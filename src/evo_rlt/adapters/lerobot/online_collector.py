from __future__ import annotations

import torch

from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_HUMAN


class RLTOnlineCollector:
    """Accumulates per-frame robot data into ChunkTransitions every C frames.

    Transitions stage locally per-episode and only commit to the global
    replay buffer on flush_episode() -- see _episode_staging.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_length: int,
        action_dim: int,
        milestone_reward: float = 0.3,
        terminal_reward: float = 1.0,
        time_decay: float = 0.995,
    ):
        self._buffer = replay_buffer
        self._C = chunk_length
        self._action_dim = action_dim
        # milestone_reward/terminal_reward are the *undiscounted* magnitudes;
        # time_decay=1.0 would reproduce the old fixed-1.0-at-the-end behavior
        # exactly, but the default here is < 1.0 (0.995 per CLOSED CHUNK, not
        # per frame -- a typical critical-phase attempt is ~50-300 chunks, so
        # 0.995 gives a real 0.6-0.8x range there instead of vanishing to ~0
        # the way a per-frame exponent over the same wall-clock span would).
        # See mark_milestone()/flush_episode() for how time_decay **
        # _chunks_closed scales the two rewards down the longer the attempt
        # takes, so faster attempts score higher without touching gamma
        # (gamma stays tuned for long-horizon TD bootstrap, not for
        # incentivizing speed -- see OnlineRLConfig.time_decay).
        self._milestone_reward = milestone_reward
        self._terminal_reward = terminal_reward
        self._time_decay = time_decay
        self._frame_actions: list[torch.Tensor] = []
        self._frame_sources: list[float] = []
        self._frame_intervention_masks: list[torch.Tensor] = []
        self._chunk_state: torch.Tensor | None = None
        self._chunk_ref: torch.Tensor | None = None
        self._chunk_is_critical: float = 0.0
        self._episode_id: int = -1
        self._prev_transition: ChunkTransition | None = None
        # Transitions accumulate here, NOT in the global replay buffer,
        # until flush_episode() commits them. A rerecorded/discarded/never-
        # labeled episode's staged transitions are simply dropped by the next
        # start_episode() call instead of permanently polluting the global
        # buffer with dangling non-terminal transitions (no valid next_state,
        # no outcome label, possibly built from footage the user rejected).
        self._episode_staging: list[ChunkTransition] = []
        # True once flush_episode() has fired for this "sub-episode" (one
        # critical-phase attempt). The recorded dataset episode may continue
        # past that point (e.g. VLA autonomously finishing a subsequent
        # placement step) -- on_frame() ignores those later frames entirely
        # so the RL reward reflects only what the actor actually controlled,
        # not whatever happens afterward. Reset by the next start_episode().
        self._flushed: bool = False
        # Chunks fully closed so far in this critical-phase attempt (does NOT
        # count the chunk currently accumulating in _frame_actions) -- the
        # clock mark_milestone()/flush_episode() decay against. Chunk
        # resolution (not per-frame): a milestone landing mid-chunk decays as
        # if it landed at that chunk's start, same ~0.33s@30fps/C=10
        # resolution mark_milestone()'s docstring already accepts for which
        # chunk the bonus lands on.
        self._chunks_closed: int = 0
        # A milestone fires at most once per attempt (see mark_milestone()).
        self._milestone_given: bool = False
        # Reward earned but not yet written into a chunk's reward_seq --
        # applied by the next _emit_transition() (routine or terminal),
        # whichever closes first. Lets mark_milestone() fire between chunk
        # boundaries without forcing an early, short chunk.
        self._pending_bonus: float = 0.0
        # Total reward (milestone + terminal) actually awarded this attempt,
        # for callers that want to log/inspect it (e.g. wandb) without
        # re-deriving it from the replay buffer.
        self.last_episode_reward: float = 0.0

    def start_episode(self, episode_id: int) -> None:
        self._episode_id = episode_id
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._frame_intervention_masks.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._prev_transition = None
        self._episode_staging = []
        self._flushed = False
        self._chunks_closed = 0
        self._milestone_given = False
        self._pending_bonus = 0.0
        self.last_episode_reward = 0.0

    def begin_attempt(self) -> None:
        """Start the critical phase on a fresh collector/chunk boundary."""
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._frame_intervention_masks.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._prev_transition = None
        self._episode_staging = []
        self._flushed = False
        self._chunks_closed = 0
        self._milestone_given = False
        self._pending_bonus = 0.0
        self.last_episode_reward = 0.0

    def cut_chunk(self) -> ChunkTransition | None:
        """Close a partial chunk before takeover/release changes its policy context."""
        if self._flushed or not self._frame_actions:
            return None
        return self._emit_transition(done=False)

    def on_frame(
        self,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
        intervention_mask: torch.Tensor | None = None,
    ) -> ChunkTransition | None:
        if self._flushed:
            return None
        if len(self._frame_actions) == 0:
            # Capture state/ref at this chunk's own start. Never borrow the
            # previous transition's tensors: after a control-source switch
            # that would turn a human correction into a mislabeled
            # (state, VLA-reference, action) tuple.
            if state_vec is not None:
                # rlt.get_last_chunk_tensors() returns compute_chunk()'s raw
                # batched tensors ((1, D) / (1, C, action_dim), B is always 1
                # for live single-robot inference) -- squeeze the leading
                # batch dim here so stored state/ref match exec_chunk's
                # unbatched (C, action_dim) shape. Without this, a chunk
                # never touched by human intervention keeps the (1, C, D)
                # shape while an intervened chunk gets (C, D) from
                # exec_chunk.clone() below, and ReplayBuffer._collate()'s
                # torch.stack() crashes the first time a batch mixes both.
                self._chunk_state = state_vec.squeeze(0) if state_vec.dim() > 1 else state_vec
                self._chunk_ref = ref_chunk.squeeze(0) if ref_chunk is not None and ref_chunk.dim() > 2 else ref_chunk
            self._chunk_is_critical = is_critical

        self._frame_actions.append(action.detach().cpu())
        self._frame_sources.append(source_type)
        if intervention_mask is None:
            intervention_mask = torch.full_like(
                action, float(source_type == SOURCE_HUMAN), dtype=torch.float32
            )
        self._frame_intervention_masks.append(intervention_mask.detach().cpu().float())

        if len(self._frame_actions) >= self._C:
            return self._emit_transition(done=False)
        return None

    def mark_milestone(self) -> float:
        """Award the one-time mid-phase shaping bonus (OnlineRLConfig.milestone_reward,
        time-decayed by chunks closed since start_episode() -- see class
        docstring). Fires at most once per critical-phase attempt: a second
        call this attempt, or any call after flush_episode(), is a no-op
        (returns 0.0). The bonus isn't written into reward_seq immediately --
        it's held in _pending_bonus and applied by whichever _emit_transition()
        closes next (a routine mid-phase chunk or the terminal one from
        flush_episode()), so pressing the milestone key mid-chunk doesn't
        force an early, short chunk boundary. Returns the (possibly
        time-decayed) bonus actually awarded, for the caller to log.
        """
        if self._flushed or self._milestone_given:
            return 0.0
        self._milestone_given = True
        bonus = self._milestone_reward * (self._time_decay ** self._chunks_closed)
        self._pending_bonus += bonus
        self.last_episode_reward += bonus
        return bonus

    def flush_episode(self, episode_success: bool) -> ChunkTransition | None:
        """Finalize the critical-phase attempt, marking the terminal
        transition done=1 and writing the sparse binary reward (terminal_reward
        iff success, else 0, both time-decayed by chunks closed by the time
        the attempt ends -- see class docstring) onto its last valid
        timestep, matching the paper's terminal-reward-only setup
        (time_decay=1.0 reproduces that exactly). Commits every transition
        staged so far to the global replay buffer, then ignores any further
        on_frame() calls until the next start_episode() -- call this at the
        moment the critical phase itself resolves (success/failure), not
        necessarily at whole-episode end: the recorded episode may continue
        afterward (e.g. VLA autonomously finishing a subsequent step), but
        that tail is dataset-only and must not leak into the RL reward. For a
        rerecorded/discarded/never-labeled attempt, just don't call this and
        let the next start_episode() drop the staged data instead.
        """
        # A duplicate UI/event callback must not relabel or reward an episode
        # that has already been committed.
        if self._flushed:
            return None

        target: ChunkTransition | None = None
        if self._frame_actions:
            # Closes the final (possibly partial) chunk -- _emit_transition()
            # increments _chunks_closed for it before we compute the decay
            # exponent below, so the terminal reward decays against a chunk
            # count that includes this very last chunk.
            target = self._emit_transition(done=True)
        elif self._prev_transition is not None:
            # Episode length was exact multiple of C — the last chunk already
            # closed (and already incremented _chunks_closed) during on_frame();
            # just mark it terminal instead of emitting a new one.
            self._prev_transition.done = torch.tensor(1.0)
            target = self._prev_transition

        terminal_reward = (
            self._terminal_reward * (self._time_decay ** self._chunks_closed) if episode_success else 0.0
        )
        self.last_episode_reward += terminal_reward
        if target is not None:
            actual = int(target.actual_steps.item())
            if self._pending_bonus != 0.0:
                target.reward_seq[actual - 1] += self._pending_bonus
                self._pending_bonus = 0.0
            target.reward_seq[actual - 1] += terminal_reward

        explicit_outcome = torch.tensor(float(episode_success))
        for transition in self._episode_staging:
            # Human corrections and autonomous actions belong to the same
            # resolved critical-phase attempt. Their source affects Actor BC
            # and sampling, not the trajectory success/failure label.
            transition.outcome = explicit_outcome.clone()
            self._buffer.add(transition)
        self._episode_staging = []
        self._flushed = True
        return target

    def _emit_transition(self, done: bool) -> ChunkTransition | None:
        if self._chunk_state is None or self._chunk_ref is None:
            self._frame_actions.clear()
            self._frame_sources.clear()
            self._frame_intervention_masks.clear()
            self._chunk_is_critical = 0.0
            return None

        actual = len(self._frame_actions)
        exec_list = self._frame_actions[: self._C]
        exec_chunk = torch.stack(exec_list)
        intervention_mask = torch.stack(self._frame_intervention_masks[: self._C])
        if actual < self._C:
            ref_for_pad = self._chunk_ref.squeeze(0) if self._chunk_ref.dim() > 2 else self._chunk_ref
            exec_chunk = torch.cat([exec_chunk, ref_for_pad[actual:self._C].to(exec_chunk)])
            intervention_mask = torch.cat(
                [
                    intervention_mask,
                    torch.zeros(
                        self._C - actual,
                        self._action_dim,
                        dtype=intervention_mask.dtype,
                    ),
                ]
            )

        # Deterministic tie-break: human wins ties (highest priority)
        dominant_source = max(set(self._frame_sources), key=lambda s: (self._frame_sources.count(s), s))
        # ref stays the true VLA reference (or its start-of-chunk fallback to
        # the last known one, above) regardless of who executed the chunk --
        # deliberately NOT exec_chunk, even for a human-dominant chunk. The
        # actor is a residual over ref (mu = ref + delta) and actor_loss's BC
        # term anchors to ref too, so ref==exec would erase exactly the
        # signal a successful human correction is most valuable for ("VLA
        # would have done X, the fix was delta=Y"), replacing it with a
        # tautological zero-delta pair the residual actor learns nothing
        # from. It also propagates: this chunk's ref becomes the *previous*
        # transition's next_ref_chunk a few lines below, so overwriting it
        # here would corrupt that transition's critic bootstrap target too,
        # not just this chunk's own BC/residual target.
        ref = self._chunk_ref.cpu()

        state = self._chunk_state.cpu()
        reward_seq = torch.zeros(self._C)
        if actual > 0 and self._pending_bonus != 0.0:
            # flush_episode() applies the terminal reward itself, after this
            # call returns (see its _chunks_closed-based decay exponent
            # computation) -- this only needs to flush any milestone bonus
            # earned since the last chunk closed.
            reward_seq[actual - 1] += self._pending_bonus
            self._pending_bonus = 0.0
        transition = ChunkTransition(
            state_vec=state,
            exec_chunk=exec_chunk,
            ref_chunk=ref,
            reward_seq=reward_seq,
            next_state_vec=state,
            next_ref_chunk=ref,
            done=torch.tensor(float(done)),
            # A single corrected element is enough to make this an
            # intervention transition; dominance is retained separately in
            # ``source`` only. This ensures brief corrections are sampled
            # from the actual chunk that contains them.
            intervention=torch.tensor(float(bool(intervention_mask[:actual].any().item()))),
            actual_steps=torch.tensor(actual),
            source=torch.tensor(int(dominant_source)),
            episode_id=torch.tensor(self._episode_id),
            is_critical=torch.tensor(self._chunk_is_critical),
            intervention_mask=intervention_mask,
        )

        if self._prev_transition is not None:
            self._prev_transition.next_state_vec = state.clone()
            self._prev_transition.next_ref_chunk = ref.clone()

        self._episode_staging.append(transition)
        self._prev_transition = transition
        self._chunks_closed += 1

        self._frame_actions.clear()
        self._frame_sources.clear()
        self._frame_intervention_masks.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        return transition
