from __future__ import annotations

import torch
import pytest


def _make_unnormalizer(action_dim: int = 4):
    """Build a real UnnormalizerProcessorStep (not a stub) with QUANTILES
    stats, matching how the SFT pi05 postprocessor is actually configured
    (see configuration_rlt_ac.py's normalization_mapping). Using the real
    class -- not a fake -- so the round-trip test below exercises the exact
    math loop.py's _normalize_executed_action relies on."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import UnnormalizerProcessorStep
    from lerobot.utils.constants import ACTION

    q01 = torch.tensor([-10.0, -20.0, -5.0, -1.0])[:action_dim]
    q99 = torch.tensor([10.0, 40.0, 5.0, 1.0])[:action_dim]
    return UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: {"q01": q01, "q99": q99}},
    )


class TestFindActionUnnormalizer:
    def test_finds_the_step_in_a_pipeline(self):
        module = pytest.importorskip("evo_rlt.adapters.lerobot.record.loop")
        step = _make_unnormalizer()

        class _FakePipeline:
            steps = [object(), step, object()]

        assert module._find_action_unnormalizer(_FakePipeline()) is step

    def test_returns_none_when_postprocessor_is_none(self):
        module = pytest.importorskip("evo_rlt.adapters.lerobot.record.loop")
        assert module._find_action_unnormalizer(None) is None

    def test_returns_none_when_no_matching_step_present(self):
        module = pytest.importorskip("evo_rlt.adapters.lerobot.record.loop")

        class _FakePipeline:
            steps = [object(), object()]

        assert module._find_action_unnormalizer(_FakePipeline()) is None


class TestNormalizeExecutedActionRoundTrip:
    def test_normalize_inverts_the_postprocessors_own_unnormalize(self):
        """The actual bug this guards against: exec_chunk recorded in
        physical robot units while ref_chunk/offline cache stay normalized.
        Round-trip through the SAME step object (unnormalize then apply our
        inverse=False call) must recover the original normalized action."""
        step = _make_unnormalizer(action_dim=4)
        normalized = torch.tensor([0.3, -0.7, 1.0, -1.0])

        physical = step._normalize_action(normalized, inverse=True)
        # Sanity: physical units are on the wide q01/q99 scale, not [-1,1].
        assert physical.abs().max().item() > 2.0

        recovered = step._normalize_action(physical, inverse=False)
        assert torch.allclose(recovered, normalized, atol=1e-5)

    def test_full_helper_matches_manual_round_trip(self):
        module = pytest.importorskip("evo_rlt.adapters.lerobot.record.loop")
        step = _make_unnormalizer(action_dim=4)

        class _FakePipeline:
            steps = [step]

        normalized = torch.tensor([0.0, 0.5, -0.25, 0.9])
        physical = step._normalize_action(normalized, inverse=True)

        # Reconstruct the closure's logic directly against the found step,
        # since _normalize_executed_action itself is a local closure inside
        # record_loop() and not separately importable.
        found = module._find_action_unnormalizer(_FakePipeline())
        recovered = found._normalize_action(physical, inverse=False)
        assert torch.allclose(recovered, normalized, atol=1e-5)
