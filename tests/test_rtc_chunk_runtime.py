from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from evo_rlt.adapters.lerobot.policies.rtc_chunk_runtime import (
    SyncRTCPolicy,
    policy_supports_rtc,
)
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

CHUNK_SIZE = 50
N_ACTION_STEPS = 10
ACTION_DIM = 12


def _rtc_config() -> RTCConfig:
    return RTCConfig(
        enabled=True,
        execution_horizon=N_ACTION_STEPS,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule("EXP"),
    )


class _FakeChunkPolicy:
    """Stands in for SmolVLA: records the RTC kwargs each chunk was asked with."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            rtc_config=None, chunk_size=CHUNK_SIZE, n_action_steps=N_ACTION_STEPS
        )
        self.calls: list[dict] = []
        self.rtc_processor_built = 0
        self.resets = 0

    def init_rtc_processor(self) -> None:
        self.rtc_processor_built += 1

    def reset(self) -> None:
        self.resets += 1

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        self.calls.append(
            {
                "prev": kwargs.get("prev_chunk_left_over"),
                "inference_delay": kwargs.get("inference_delay"),
                "batch_keys": sorted(batch),
            }
        )
        return torch.arange(
            len(self.calls) * 1000, len(self.calls) * 1000 + CHUNK_SIZE * ACTION_DIM, dtype=torch.float32
        ).reshape(1, CHUNK_SIZE, ACTION_DIM)


def _runtime() -> tuple[SyncRTCPolicy, _FakeChunkPolicy]:
    policy = _FakeChunkPolicy()
    return SyncRTCPolicy(policy, _rtc_config(), n_action_steps=N_ACTION_STEPS), policy


def _batch() -> dict[str, torch.Tensor]:
    return {"observation.state": torch.zeros(1, ACTION_DIM)}


def test_rtc_is_switched_on_in_the_policy_config() -> None:
    """RTC lives in the model's denoising loop and reads config.rtc_config.

    Setting the config without rebuilding the processor leaves the model with
    the old (absent) one, and chunks come back unguided with no error.
    """
    runtime, policy = _runtime()
    assert policy.config.rtc_config.enabled is True
    assert policy.rtc_processor_built == 1


def test_replans_at_the_same_cadence_as_plain_chunking() -> None:
    """One inference per n_action_steps, as without RTC -- the point is that the
    chunk is guided, not that it is recomputed more often."""
    runtime, policy = _runtime()
    for _ in range(N_ACTION_STEPS * 3):
        runtime.select_action(_batch())
    assert len(policy.calls) == 3


def test_refills_while_actions_remain_so_rtc_has_a_prefix() -> None:
    """Waiting for an empty queue would hand RTC an empty prefix.

    prev_chunk_left_over is what constrains the new chunk to continue the old
    one; without it each chunk is an independent draw and the seam is exactly
    the jump RTC exists to remove. First call has no previous chunk, which
    RTCProcessor handles by skipping guidance.
    """
    runtime, policy = _runtime()
    for _ in range(N_ACTION_STEPS + 1):
        runtime.select_action(_batch())
    assert len(policy.calls) == 2
    assert policy.calls[0]["prev"] is None
    prev = policy.calls[1]["prev"]
    assert prev is not None and prev.shape[0] == CHUNK_SIZE - N_ACTION_STEPS


def test_inference_delay_is_zero_because_the_loop_blocks() -> None:
    """The control loop blocks on inference and the simulator advances a fixed
    1/fps per step, so nothing executes while a chunk is computed. A non-zero
    delay would make RTC drop leading actions that were never executed."""
    runtime, policy = _runtime()
    for _ in range(N_ACTION_STEPS + 1):
        runtime.select_action(_batch())
    assert {call["inference_delay"] for call in policy.calls} == {0}


def test_actions_are_served_in_order_from_the_chunk() -> None:
    runtime, policy = _runtime()
    served = torch.cat([runtime.select_action(_batch()) for _ in range(N_ACTION_STEPS)])
    expected = policy.predict_action_chunk(_batch()).squeeze(0)[:N_ACTION_STEPS]
    # predict_action_chunk above is call #2; compare against the first chunk.
    first = torch.arange(1000, 1000 + CHUNK_SIZE * ACTION_DIM, dtype=torch.float32).reshape(
        CHUNK_SIZE, ACTION_DIM
    )
    assert torch.equal(served, first[:N_ACTION_STEPS])
    assert expected.shape == (N_ACTION_STEPS, ACTION_DIM)


def test_reset_clears_the_queue_and_the_policy() -> None:
    runtime, policy = _runtime()
    runtime.select_action(_batch())
    runtime.reset()
    runtime.select_action(_batch())
    assert policy.resets == 1
    assert len(policy.calls) == 2
    assert policy.calls[1]["prev"] is None  # nothing carried across an episode


def test_runs_under_inference_mode() -> None:
    """The real caller wraps inference in torch.inference_mode().

    RTC guidance differentiates through the denoiser, and inference mode is
    stronger than no_grad: enable_grad() cannot lift it, and tensors created
    under it are permanently barred from autograd. Without leaving inference
    mode, the first guided chunk dies with "element 0 of tensors does not
    require grad" -- during a rollout, not in any preflight.
    """
    runtime, policy = _runtime()
    with torch.inference_mode():
        for _ in range(N_ACTION_STEPS + 1):
            runtime.select_action(_batch())
    prev = policy.calls[1]["prev"]
    assert prev is not None
    assert not prev.is_inference(), "prefix is still an inference tensor; autograd will refuse it"


def test_wrapper_proxies_the_underlying_policy() -> None:
    runtime, policy = _runtime()
    assert runtime.config is policy.config
    assert runtime.unwrapped is policy


def test_act_is_not_wrappable() -> None:
    """ACT has no rtc_config and nothing to inpaint -- it is not a flow policy."""
    act_like = SimpleNamespace(config=SimpleNamespace(chunk_size=100), predict_action_chunk=lambda b: b)
    assert not policy_supports_rtc(act_like)
    with pytest.raises(TypeError, match="cannot be driven by chunk-level RTC"):
        SyncRTCPolicy(act_like, _rtc_config(), n_action_steps=N_ACTION_STEPS)


def test_backend_wraps_the_policy_before_the_record_loop() -> None:
    """_maybe_wrap_with_rtc has to be applied to the policy record_loop uses.

    Wrapping a policy the loop never sees leaves the rollout on plain chunking
    while every log line still says RTC is enabled.
    """
    from evo_rlt.adapters.lerobot.record import backend

    tree = ast.parse(inspect.getsource(backend))
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_maybe_wrap_with_rtc"
    ]
    assert assigns, "_maybe_wrap_with_rtc's result is discarded; the wrapper never reaches record_loop"
    for node in assigns:
        assert any(isinstance(t, ast.Name) and t.id == "policy" for t in node.targets)
