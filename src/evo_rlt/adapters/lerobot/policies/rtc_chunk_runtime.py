"""Real-Time Chunking for policies whose `select_action` refuses to do it.

SmolVLA implements RTC where it matters -- `sample_actions()` routes every
denoising step through `RTCProcessor.denoise_step()` when a chunk arrives with
`prev_chunk_left_over` -- but `select_action()` asserts RTC is *off*, because
the action queue it keeps internally has no notion of a previous chunk to be
guided by. RTC is only reachable through `predict_action_chunk()`, and that
leaves the queue to the caller. This module is that caller.

Why it is worth the wiring: without RTC each chunk is an independent draw from
the flow-matching prior, and on this rig those draws are far apart -- two
samples of the same observation differ about as much as either differs from the
demonstration. Re-planning every 10 steps therefore swaps trajectories mid-motion,
which is what shows up as the arm stepping forward and back. Measured on the 60k
checkpoint, the step at a chunk boundary is 2.6x the steps inside a chunk. RTC
makes the new chunk an inpainting of the actions still queued from the old one,
so the seam is constrained instead of resampled.

Synchronous on purpose. `record_loop` blocks on inference and the simulator
advances a fixed 1/fps per control step, so no wall-clock time passes in the
simulation while a chunk is being computed: the robot cannot drift during
inference and `inference_delay` is genuinely 0. The asynchronous RTC runtime in
`modeling_rlt_ac.py` exists because pi0.5 inference is slow enough to matter on
real hardware; none of that machinery buys anything here, and all of it could
mask a bug.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def policy_supports_rtc(policy: Any) -> bool:
    """Whether this policy can be driven by chunk-level RTC.

    Needs both halves: a config slot RTC reads (`rtc_config`), and a
    `predict_action_chunk` that forwards RTC kwargs into the denoiser. ACT has
    neither -- it is not a diffusion/flow policy and has nothing to inpaint.
    """
    return hasattr(policy, "config") and hasattr(policy.config, "rtc_config") and hasattr(
        policy, "predict_action_chunk"
    )


class SyncRTCPolicy:
    """Wrap a chunk policy so `select_action()` serves RTC-guided chunks.

    Proxies everything it does not define, so callers that read `policy.config`,
    call `policy.eval()`, or duck-type for `set_rl_mode` keep working on the
    wrapped policy unchanged.
    """

    def __init__(
        self,
        policy: Any,
        rtc_config: RTCConfig,
        *,
        n_action_steps: int,
        refill_threshold: int | None = None,
    ) -> None:
        if not policy_supports_rtc(policy):
            raise TypeError(f"{type(policy).__name__} cannot be driven by chunk-level RTC")
        chunk_size = int(policy.config.chunk_size)
        if n_action_steps < 1 or n_action_steps > chunk_size:
            raise ValueError(f"n_action_steps must be in [1, {chunk_size}], got {n_action_steps}")

        self._policy = policy
        self._rtc_config = rtc_config
        self._n_action_steps = n_action_steps
        # Re-plan once every n_action_steps, the same cadence as without RTC --
        # but do it while actions are still queued, because those leftovers are
        # exactly what guides the next chunk. Waiting for an empty queue would
        # hand RTC an empty prefix and degrade it back to an independent draw.
        self._refill_threshold = (
            chunk_size - n_action_steps if refill_threshold is None else refill_threshold
        )
        self._queue = ActionQueue(rtc_config)

        # RTC lives in the model's denoising loop and is switched on by the
        # config the processor was built from, so both have to be set before
        # the first chunk. init_rtc_processor() re-runs after the model exists
        # and pushes the processor down into it.
        policy.config.rtc_config = rtc_config
        policy.init_rtc_processor()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def unwrapped(self) -> Any:
        return self._policy

    def reset(self) -> None:
        self._queue.clear()
        self._policy.reset()

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if self._queue.qsize() <= self._refill_threshold:
            self._refill(batch)
        action = self._queue.get()
        if action is None:  # refill produced nothing usable
            raise RuntimeError("RTC action queue is empty right after a refill")
        return action.unsqueeze(0)

    def _refill(self, batch: dict[str, torch.Tensor]) -> None:
        action_index_before = self._queue.get_action_index()
        # RTC guidance differentiates the denoised chunk w.r.t. the latent
        # (`torch.autograd.grad` inside RTCProcessor.denoise_step), and the
        # caller runs inference under torch.inference_mode(). Inference mode is
        # stronger than no_grad: the enable_grad() already inside denoise_step
        # cannot lift it, and tensors created under it are permanently barred
        # from autograd. So leave inference mode for the chunk call and clone
        # every tensor crossing in, which is what strips the inference flag.
        with torch.inference_mode(False), torch.enable_grad():
            prev_chunk_left_over = self._queue.get_left_over()
            if prev_chunk_left_over is not None:
                prev_chunk_left_over = prev_chunk_left_over.clone()
            batch = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            chunk = self._refill_chunk(batch, prev_chunk_left_over)

        actions = chunk.squeeze(0).detach()
        real_delay = max(0, self._queue.get_action_index() - action_index_before)
        # With rtc_config.enabled, merge() *replaces* the queue with the new
        # chunk rather than appending -- the new chunk already accounts for the
        # old one through prefix attention, so keeping both would double up.
        self._queue.merge(actions, actions, real_delay, action_index_before)

    def _refill_chunk(
        self, batch: dict[str, torch.Tensor], prev_chunk_left_over: torch.Tensor | None
    ) -> torch.Tensor:
        chunk = self._policy.predict_action_chunk(
            batch,
            # 0, and measured rather than assumed: the queue index below cannot
            # have moved, because this call blocks the control loop. Passing a
            # non-zero delay would make RTC discard leading actions that were
            # never executed.
            inference_delay=0,
            # None on the first chunk, which is exactly right: RTCProcessor
            # skips guidance entirely when there is no prefix to inpaint against.
            prev_chunk_left_over=prev_chunk_left_over,
            # Left unset so the horizon has one source, rtc_config.execution_horizon.
        )
        if chunk.shape[0] != 1:
            raise ValueError(f"RTC deployment expects batch size 1, got {chunk.shape[0]}")
        return chunk
