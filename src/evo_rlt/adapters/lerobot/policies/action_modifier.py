from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.phase_controller import PhaseController
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.utils import (
    flatten_chunk,
    postprocess_prefix_tokens,
    project_action_delta,
    unflatten_chunk,
)


class PrefixOutputCapture:
    """Capture prefix hidden states from PI05Policy's PaliGemmaWithExpertModel.

    PI05's ``sample_actions`` calls ``paligemma_with_expert.forward()``
    directly (not via ``__call__``), so standard PyTorch forward hooks
    never fire.  Instead we monkey-patch ``forward`` to intercept the
    prefix-only call (``inputs_embeds=[prefix_embs, None]``).

    After capture the raw prefix tokens are optionally sliced to image-only
    (dropping language tokens) and pooled from (B, ~968, 2048) to
    (B, token_pool_size, 2048) via adaptive average pooling.
    """

    def __init__(
        self,
        token_pool_size: int = 64,
        image_only: bool = False,
        num_image_tokens: int = 0,
    ):
        self.token_pool_size = token_pool_size
        self.image_only = image_only
        self.num_image_tokens = num_image_tokens
        self._captured: Tensor | None = None
        self._original_forward = None
        self._target = None

    def attach(self, policy) -> None:
        """Monkey-patch ``forward`` on ``policy.model.paligemma_with_expert``.

        When image_only is true and num_image_tokens is unset (0), derive it from
        the SigLIP vision config so callers don't need to know paligemma internals.
        """
        target = policy.model.paligemma_with_expert
        self._target = target
        self._original_forward = target.forward

        if self.image_only and self.num_image_tokens == 0:
            self.num_image_tokens = self._infer_num_image_tokens(policy)

        capture = self  # closure reference

        def patched_forward(*args, **kwargs):
            result = capture._original_forward(*args, **kwargs)
            outputs, _past_kv = result
            prefix_tokens = outputs[0]
            if prefix_tokens is not None:
                capture._captured = postprocess_prefix_tokens(
                    prefix_tokens.detach().float(),
                    image_only=capture.image_only,
                    num_image_tokens=capture.num_image_tokens,
                    pool_size=capture.token_pool_size,
                )
            return result

        target.forward = patched_forward

    @staticmethod
    def _infer_num_image_tokens(policy) -> int:
        pi05_cfg = policy.config
        vision_cfg = policy.model.paligemma_with_expert.paligemma.config.vision_config
        n_per_cam = (pi05_cfg.image_resolution[0] // vision_cfg.patch_size) ** 2
        return n_per_cam * len(pi05_cfg.image_features)

    def consume(self) -> Tensor:
        """Return and clear the captured prefix tokens.

        Raises AssertionError if no prefix output has been captured yet (i.e.
        the VLA forward pass has not run since the last consume).
        """
        assert self._captured is not None, (
            "No prefix_output captured -- VLA forward not yet called"
        )
        result = self._captured
        self._captured = None
        return result

    def detach(self) -> None:
        """Restore the original forward method."""
        if self._original_forward is not None and self._target is not None:
            self._target.forward = self._original_forward
            self._original_forward = None
            self._target = None


@dataclass
class RLTStepMetadata:
    """Per-step metadata emitted alongside each popped action."""

    phase: float  # 0.0 = VLA, 1.0 = critical/RL
    source_type: float  # 0.0 = VLA action, 1.0 = RL action


class RLTActionModifier(nn.Module):
    """RL Token Encoder + Actor + Phase Controller + Chunk Queue.

    Sits between the VLA action output and the final action used by the robot.
    In VLA phase the VLA chunk is passed through unchanged; in RL phase the
    Actor refines the chunk conditioned on the RL-token state representation.
    """

    def __init__(
        self,
        rl_token: RLTokenModule,
        actor: ChunkActor,
        phase_ctrl: PhaseController,
        chunk_length: int,
        action_dim: int,
        proprio_dim: int,
        chunk_exec_steps: int = 25,
        vla_ref: bool = True,
        action_clip_delta: float | None = None,
        slew_rate_limit: float | None = None,
        actor_deploy_scale: float = 1.0,
    ):
        super().__init__()
        self.rl_token = rl_token
        self.actor = actor
        self.phase_ctrl = phase_ctrl
        self.chunk_length = chunk_length
        self.chunk_exec_steps = chunk_exec_steps
        self.vla_ref = vla_ref
        self.action_clip_delta = action_clip_delta
        self.slew_rate_limit = slew_rate_limit
        self.set_actor_deploy_scale(actor_deploy_scale)
        self._cc_log_count = 0  # throttle for the compute_chunk ref diagnostic
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self._action_queue: deque[Tensor] = deque()
        # Actor residuals aligned one-to-one with _action_queue.  Keeping a
        # parallel queue lets pop_action() remember the residual of the frame
        # that was actually dispatched, rather than the tail of a chunk that
        # may later be interrupted and never executed.
        self._residual_queue: deque[Tensor] = deque()
        self._pending_residual_chunk: Tensor | None = None
        self._step_metadata: deque[RLTStepMetadata] = deque()
        # Last computed RL-phase chunk tensors, for online-RL transition
        # collection (see RLTOnlineCollector). Populated only in RL phase;
        # persists (peek, not pop) until overwritten by the next compute_chunk()
        # or cleared by reset() -- see get_last_chunk_tensors().
        self._last_state_vec: Tensor | None = None
        self._last_ref_chunk: Tensor | None = None
        # The most recent actor residual that was actually popped.  This is
        # deliberately not inferred from the absolute executed action: doing
        # so would mix VLA reference motion (or a human correction) into the
        # quantity the actor-residual limiter owns.
        self._last_actor_residual: Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_rl_phase(self) -> bool:
        return self.phase_ctrl.is_critical

    @property
    def needs_new_chunk(self) -> bool:
        return len(self._action_queue) == 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_chunk(
        self,
        vla_chunk: Tensor,
        proprio: Tensor,
        prefix_tokens: Tensor,
    ) -> Tensor:
        """Compute action chunk, either VLA pass-through or RL-refined.

        Args:
            vla_chunk: (B, H, action_dim) normalised VLA action chunk.
            proprio: (B, proprio_dim) normalised proprioceptive state.
            prefix_tokens: (B, pool_size, token_dim) pooled VLA prefix tokens.

        Returns:
            VLA phase:  (B, chunk_exec_steps, action_dim)
            RL phase:   (B, chunk_length, action_dim). Within
                action_clip_delta of ref_chunk if action_clip_delta is set
                (see project_action_delta), else clamped to [-1, 1].
        """
        phase_val = 1.0 if self.is_rl_phase else 0.0
        source_val = phase_val

        if not self.is_rl_phase:
            # VLA phase: take the first chunk_exec_steps consecutive actions.
            # Not slew-limited (VLA output is trusted as-is).  Its actor
            # residual is exactly zero, so a later VLA->RL handoff starts
            # from the correct residual anchor without rate-limiting the VLA
            # trajectory itself.
            n = min(self.chunk_exec_steps, vla_chunk.shape[1])
            out = vla_chunk[:, :n, :]
            self._pending_residual_chunk = torch.zeros_like(out)
            self._enqueue_metadata(phase_val, source_val, n)
            return out

        # RL phase: take first chunk_length frames as actor reference
        ref_chunk = vla_chunk[:, :self.chunk_length, :]
        z_rl = self.rl_token.encode(prefix_tokens)
        state_vec = torch.cat([z_rl, proprio], dim=-1)
        ref_flat = flatten_chunk(ref_chunk)
        if not self.vla_ref:
            # Hide the VLA reference from the actor: training ref-dropout
            # multiplies ref_chunk_flat by a 0/1 mask, so a dropped sample
            # is exactly an all-zero ref. Reproduce that here.
            ref_flat = torch.zeros_like(ref_flat)
        should_log = self._cc_log_count < 3 or self._cc_log_count % 30 == 0
        mu, _ = self.actor(state_vec, ref_flat, training=False)
        mu_chunk = unflatten_chunk(mu, self.chunk_length)
        # Training and physical control are deliberately decoupled for safe
        # online warmup.  The Actor may already be learning demonstrations,
        # but scale=0 makes the critical-phase command an exact VLA
        # pass-through.  Only after critic-only warmup does OnlineRLTrainer
        # ramp this residual contribution toward 1.
        if self.actor_deploy_scale == 0.0:
            chunk = ref_chunk
        else:
            mu_chunk = ref_chunk + self.actor_deploy_scale * (mu_chunk - ref_chunk)
            if self.action_clip_delta is not None:
                # Safety bound: limit how far the (possibly still-training)
                # RL actor's output may deviate from the VLA reference chunk.
                # The same projection is used in actor_loss/critic_loss so
                # training sees the action that actually gets executed.
                chunk = project_action_delta(mu_chunk, ref_chunk, self.action_clip_delta)
            else:
                chunk = mu_chunk.clamp(-1, 1)
        if self.slew_rate_limit is not None and self.actor_deploy_scale > 0.0:
            # Runtime bound on how quickly the learned contribution can move
            # within the ref-relative delta range.  The trusted VLA reference
            # itself remains untouched. Target policy smoothing (see
            # critic_loss's target_noise_std) only smooths the critic's
            # local Q-estimate; it does not make the actor residual
            # temporally continuous. See actor_loss's
            # smoothness_weight for a training-time complement to this.
            chunk = self._apply_slew_rate_limit(chunk, ref_chunk)
        self._last_state_vec = state_vec.detach().clone()
        self._last_ref_chunk = ref_chunk.detach().clone()
        if should_log:
            delta = (chunk - ref_chunk).abs()
            # Diagnostic: the VLA ref is only the actor input; the returned
            # chunk below is the RLT actor output that will be executed.
            print(
                f"[RLT source=RLT_ACTOR vla_ref={self.vla_ref}] "
                f"compute_chunk #{self._cc_log_count}: "
                f"actor_deploy_scale={self.actor_deploy_scale:.3f}, "
                f"vla_ref[0,0,:4]={[round(v, 4) for v in ref_chunk[0, 0, :4].tolist()]}, "
                f"actor_out[0,0,:4]={[round(v, 4) for v in chunk[0, 0, :4].tolist()]}, "
                f"mean_abs_delta={delta.mean().item():.4f}, "
                f"max_abs_delta={delta.max().item():.4f}, "
                f"ref-into-actor abs-sum={ref_flat.abs().sum().item():.4f}",
                flush=True,
            )
        self._cc_log_count += 1

        # enqueue()/pop_action() carry these residuals in lockstep with the
        # returned actions.  Do not update _last_actor_residual here: this
        # chunk may be interrupted before all (or any) of it is dispatched.
        self._pending_residual_chunk = (chunk - ref_chunk).detach().clone()
        self._enqueue_metadata(phase_val, source_val, self.chunk_length)
        return chunk

    def set_actor_deploy_scale(self, scale: float) -> None:
        """Set the fraction of the Actor residual allowed onto hardware.

        ``0`` is an exact VLA pass-through and ``1`` is the full Actor
        policy.  Clearing a queued chunk makes a changed scale effective at
        the next inference rather than leaking commands computed under the
        previous scale.
        """
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError("actor_deploy_scale must be in [0, 1]")
        if hasattr(self, "actor_deploy_scale") and scale != self.actor_deploy_scale:
            self.interrupt_chunk()
        self.actor_deploy_scale = scale

    def _apply_slew_rate_limit(self, chunk: Tensor, ref_chunk: Tensor) -> Tensor:
        """Sequentially clamp how fast the ACTOR RESIDUAL (chunk - ref_chunk)
        may change from one physical timestep to the next.  pop_action()
        carries the last residual that was really dispatched, so interrupted
        chunks cannot leak unexecuted tail frames into the next chunk.

        Limiting the absolute command would also rate-limit the trusted VLA
        reference, distort gripper timing, and allow reference-relative lag
        to exceed action_clip_delta.  Limiting only the residual keeps the
        intended property (the actor contribution cannot jump) while leaving
        the reference untouched.
        Absolute continuity across an intervention release is a separate
        concern owned by loop.py's release action blend.  Human actions are
        therefore never converted into actor residuals.  Hard clamp is fine
        here (no gradient needed: compute_chunk runs under @torch.no_grad()).
        """
        residual = chunk - ref_chunk
        if self._last_actor_residual is None:
            # Nothing dispatched yet (fresh reset / first frame): ramp the
            # residual in from "actor contributes nothing", the safe default.
            prev = torch.zeros_like(residual[:, 0, :])
        else:
            prev = self._last_actor_residual.to(
                device=chunk.device, dtype=chunk.dtype
            )
        limited_steps = []
        for t in range(residual.shape[1]):
            prev = prev + (residual[:, t, :] - prev).clamp(
                -self.slew_rate_limit, self.slew_rate_limit
            )
            limited_steps.append(prev)
        return ref_chunk + torch.stack(limited_steps, dim=1)

    def _enqueue_metadata(self, phase: float, source: float, count: int) -> None:
        """Enqueue metadata entries for every step in the upcoming chunk."""
        for _ in range(count):
            self._step_metadata.append(RLTStepMetadata(phase=phase, source_type=source))

    def enqueue(self, chunk: Tensor) -> None:
        """Enqueue chunk steps and their aligned actor residuals.

        Args:
            chunk: (B, C, action_dim).
        """
        residual_chunk = self._pending_residual_chunk
        if residual_chunk is None or residual_chunk.shape != chunk.shape:
            raise RuntimeError(
                "enqueue() must immediately follow compute_chunk() with the returned chunk"
            )
        self._action_queue.extend(chunk.transpose(0, 1))
        self._residual_queue.extend(residual_chunk.transpose(0, 1))
        self._pending_residual_chunk = None

    def pop_action(self) -> Tensor:
        """Pop and return the next single-step action from the queue, and
        record its aligned actor residual as the slew-rate continuity
        reference.  This, not compute_chunk(), is the moment a frame is
        actually selected for dispatch."""
        action = self._action_queue.popleft()
        residual = self._residual_queue.popleft()
        self._last_actor_residual = residual.detach().clone()
        return action

    def pop_step_metadata(self) -> RLTStepMetadata | None:
        """Pop and return the next step's metadata, or None if empty."""
        if len(self._step_metadata) == 0:
            return None
        return self._step_metadata.popleft()

    def get_last_chunk_tensors(self) -> tuple[Tensor, Tensor] | None:
        """Peek (not pop) the (state_vec, ref_chunk) from the most recently
        computed RL-phase chunk, or None if none has been computed yet.

        Deliberately non-consuming: the recording loop reads it on every
        physical frame, while it changes only when a new actor chunk is
        computed. This keeps all frames in that chunk attached to the same
        state/reference without removing the cache after its first read.

        Does NOT include the actor's output chunk: RLTOnlineCollector builds
        exec_chunk itself from the actual per-frame executed actions (which
        may differ from the actor's raw output under human intervention), not
        from what the actor originally proposed.
        """
        if self._last_state_vec is None:
            return None
        return (self._last_state_vec, self._last_ref_chunk)

    # ------------------------------------------------------------------
    # Phase control (duck-typed interface for recording_loop)
    # ------------------------------------------------------------------

    def set_rl_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_critical()

    def set_vla_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_vla()

    def trigger_critical_phase(self) -> None:
        """Toggle between VLA and critical phase, clearing queues."""
        self.interrupt_chunk()
        if self.phase_ctrl.is_critical:
            self.phase_ctrl.trigger_vla()
        else:
            self.phase_ctrl.trigger_critical()

    def interrupt_chunk(self) -> None:
        """Clear action and metadata queues (e.g. on phase switch or human
        intervention start). Deliberately does NOT clear the last-chunk
        tensor cache (see get_last_chunk_tensors); the next inference replaces
        it with a fresh encoding, including counterfactual inference performed
        while a human action is selected for execution."""
        self._action_queue.clear()
        self._residual_queue.clear()
        self._pending_residual_chunk = None
        self._step_metadata.clear()

    def reset(self) -> None:
        """Reset queues and phase controller to initial state."""
        self._action_queue.clear()
        self._residual_queue.clear()
        self._pending_residual_chunk = None
        self._step_metadata.clear()
        self._last_state_vec = None
        self._last_ref_chunk = None
        self._last_actor_residual = None
        self.phase_ctrl.reset()
