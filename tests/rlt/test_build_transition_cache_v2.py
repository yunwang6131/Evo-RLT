from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


def test_left_only_exec_chunk_uses_demo_left_and_vla_right():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    demonstrated = torch.arange(24, dtype=torch.float32).view(1, 2, 12)
    reference = -torch.arange(24, dtype=torch.float32).view(1, 2, 12)

    executed = module._compose_exec_chunk(demonstrated, reference, "left")

    assert torch.equal(executed[..., :6], demonstrated[..., :6])
    assert torch.equal(executed[..., 6:], reference[..., 6:])
    # Helper must not mutate either input cache tensor.
    assert torch.equal(demonstrated, torch.arange(24, dtype=torch.float32).view(1, 2, 12))


def test_exec_chunk_preserves_real_demonstrated_action_outside_actor_bound():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    demonstrated = torch.tensor([[[-2.0, -0.05, 0.05, 2.0]]])
    reference = torch.zeros_like(demonstrated)

    executed = module._compose_exec_chunk(demonstrated, reference, "both")

    # The critic's behavior action must be the one that actually earned the
    # outcome. Actor bounds are applied to actor candidates, not by inventing
    # a different successful behavior transition in the cache.
    assert torch.equal(executed, demonstrated)


@pytest.mark.parametrize(
    ("rl_action_arms", "expected"),
    [
        ("both", [[1.0, 1.0, 1.0, 1.0]]),
        ("left", [[1.0, 1.0, 0.0, 0.0]]),
    ],
)
def test_successful_demo_supervises_actor_controlled_dimensions(rl_action_arms, expected):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    exec_chunk = torch.zeros(1, 4)

    mask = module._demonstration_supervision_mask(
        exec_chunk, rl_action_arms, episode_success=True
    )

    assert torch.equal(mask, torch.tensor(expected))


def test_failed_demo_does_not_directly_supervise_actor():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    exec_chunk = torch.zeros(2, 4)

    mask = module._demonstration_supervision_mask(
        exec_chunk, "both", episode_success=False
    )

    assert torch.equal(mask, torch.zeros_like(exec_chunk))


def test_missing_demo_outcome_can_default_to_success():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    dataset = SimpleNamespace(meta=SimpleNamespace(episodes={}))
    assert module._get_episode_success(dataset, 0, default_outcome="success") is True


def test_episode_critical_bounds_converts_local_segment_to_absolute():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    # Episode 7 starts at absolute row 1000; segment [50, 80, "success"] is
    # local to the episode -- end is inclusive per the labeler's convention.
    labels = {"7": {"segment": [50, 80, "success"], "no_critical": False}}
    assert module._episode_critical_bounds(labels, ep_id=7, ep_from=1000) == (
        1050,
        1081,
        True,
        None,
    )


def test_episode_critical_bounds_failure_outcome():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    labels = {"3": {"segment": [10, 20, "failure"], "no_critical": False}}
    assert module._episode_critical_bounds(labels, ep_id=3, ep_from=0) == (
        10,
        21,
        False,
        None,
    )


def test_episode_critical_bounds_converts_local_milestone_to_absolute():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    labels = {"7": {"segment": [50, 80, "success"], "no_critical": False, "milestone_frame": 65}}
    assert module._episode_critical_bounds(labels, ep_id=7, ep_from=1000) == (1050, 1081, True, 1065)


@pytest.mark.parametrize(
    "entry",
    [
        None,  # no label at all for this episode
        {"segment": None, "no_critical": False},
        {"segment": [1, 2, "success"], "no_critical": True},
    ],
)
def test_episode_critical_bounds_skips_unlabeled_or_no_critical(entry):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    labels = {} if entry is None else {"9": entry}
    assert module._episode_critical_bounds(labels, ep_id=9, ep_from=0) is None


class TestResolveMilestoneChunk:
    def test_no_milestone_returns_none(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        result = module._resolve_milestone_chunk(
            None, critical_start_frame=0, episode_last_frame=99,
            chunk_length=10, frame_indices=list(range(0, 100, 2)), ep_id=0,
        )
        assert result == (None, None)

    def test_exact_anchor_when_stride_divides_chunk_length(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        # milestone at local frame 25 (offset 25 from segment start=0) with
        # C=10 -> 2 chunks closed before it (frames 0-9, 10-19), landing in
        # chunk [20, 30) -> bonus deposited on the anchor starting at 20.
        frame_indices = list(range(0, 100, 2))
        start, chunks_closed = module._resolve_milestone_chunk(
            25, critical_start_frame=0, episode_last_frame=99,
            chunk_length=10, frame_indices=frame_indices, ep_id=0,
        )
        assert (start, chunks_closed) == (20, 2)

    def test_milestone_on_chunk_boundary(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        # milestone exactly at the start of a chunk (frame 20): still 2
        # chunks closed before it (0-9, 10-19), same as landing anywhere in
        # [20, 30).
        frame_indices = list(range(0, 100, 2))
        start, chunks_closed = module._resolve_milestone_chunk(
            20, critical_start_frame=0, episode_last_frame=99,
            chunk_length=10, frame_indices=frame_indices, ep_id=0,
        )
        assert (start, chunks_closed) == (20, 2)

    def test_out_of_range_milestone_raises(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        with pytest.raises(ValueError, match="outside its own critical segment"):
            module._resolve_milestone_chunk(
                150, critical_start_frame=0, episode_last_frame=99,
                chunk_length=10, frame_indices=list(range(0, 100, 2)), ep_id=0,
            )

    def test_no_usable_anchor_returns_none(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        # Segment shorter than chunk_length: no anchor can ever produce a
        # full C-length transition, so the bonus has nowhere to land.
        start, chunks_closed = module._resolve_milestone_chunk(
            3, critical_start_frame=0, episode_last_frame=5,
            chunk_length=10, frame_indices=[0, 2, 4], ep_id=0,
        )
        assert (start, chunks_closed) == (None, None)

    def test_misaligned_stride_falls_back_to_nearest_anchor(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        # stride=3 does not divide chunk_length=10, so the exact target start
        # frame (20) is not guaranteed to be an anchor -- here frame_indices
        # (stride 3 from 0) never includes 20 exactly; nearest is 21.
        frame_indices = [f for f in range(0, 100, 3)]
        start, chunks_closed = module._resolve_milestone_chunk(
            25, critical_start_frame=0, episode_last_frame=99,
            chunk_length=10, frame_indices=frame_indices, ep_id=0,
        )
        assert chunks_closed == 2
        assert start in frame_indices
        assert abs(start - 20) <= 1  # nearest anchor to the exact target (20)


class TestComputeChunkReward:
    def test_terminal_only(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=90, is_last=True, episode_success=True,
            terminal_chunks_closed=10, milestone_actual_start_frame=None,
            milestone_chunks_closed=None, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995,
        )
        expected = torch.zeros(10)
        expected[-1] = 1.0 * (0.995 ** 10)
        assert torch.allclose(reward, expected)

    def test_no_terminal_reward_on_failure(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=90, is_last=True, episode_success=False,
            terminal_chunks_closed=10, milestone_actual_start_frame=None,
            milestone_chunks_closed=None, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995,
        )
        assert torch.equal(reward, torch.zeros(10))

    def test_milestone_only_on_non_terminal_chunk(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=20, is_last=False, episode_success=True,
            terminal_chunks_closed=10, milestone_actual_start_frame=20,
            milestone_chunks_closed=2, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995,
        )
        expected = torch.zeros(10)
        expected[-1] = 0.3 * (0.995 ** 2)
        assert torch.allclose(reward, expected)

    def test_milestone_awarded_even_on_episode_failure(self):
        """Matches RLTOnlineCollector.mark_milestone(): the bonus doesn't
        care about the eventual success/failure outcome."""
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=20, is_last=False, episode_success=False,
            terminal_chunks_closed=10, milestone_actual_start_frame=20,
            milestone_chunks_closed=2, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995,
        )
        expected = torch.zeros(10)
        expected[-1] = 0.3 * (0.995 ** 2)
        assert torch.allclose(reward, expected)

    def test_milestone_and_terminal_combine_on_same_chunk(self):
        """When the milestone lands in the segment's final chunk, both
        bonuses land on the same reward_seq slot -- matching
        RLTOnlineCollector.flush_episode()'s combined pending_bonus +
        terminal_reward add."""
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=90, is_last=True, episode_success=True,
            terminal_chunks_closed=10, milestone_actual_start_frame=90,
            milestone_chunks_closed=9, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995,
        )
        expected = torch.zeros(10)
        expected[-1] = 1.0 * (0.995 ** 10) + 0.3 * (0.995 ** 9)
        assert torch.allclose(reward, expected)

    def test_time_decay_one_reproduces_old_fixed_reward_behavior(self):
        module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
        reward = module._compute_chunk_reward(
            start_frame=90, is_last=True, episode_success=True,
            terminal_chunks_closed=37, milestone_actual_start_frame=None,
            milestone_chunks_closed=None, chunk_length=10,
            milestone_reward=0.3, terminal_reward=1.0, time_decay=1.0,
        )
        expected = torch.zeros(10)
        expected[-1] = 1.0
        assert torch.allclose(reward, expected)


def test_load_critical_segments_missing_file_returns_empty(tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    assert module._load_critical_segments(tmp_path / "meta" / "critical_segments.json") == {}
    assert module._load_critical_segments(None) == {}


def test_load_critical_segments_reads_labeler_json(tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    import json

    path = tmp_path / "critical_segments.json"
    path.write_text(json.dumps({"0": {"segment": [5, 15, "success"], "no_critical": False}}))
    labels = module._load_critical_segments(path)
    assert labels == {"0": {"segment": [5, 15, "success"], "no_critical": False}}


def test_transition_cache_v2_passes_video_backend(monkeypatch, tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_episodes = 0
            self.meta = SimpleNamespace(episodes=None)

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=12,
            chunk_size=50,
            image_only=False,
            proprio_dim=12,
            token_pool_size=0,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 0
        _pi05 = object()
        rl_token = object()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def attach(self, pi05):
            pass

        def detach(self):
            pass

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module.RLTokenPolicy, "from_pretrained", lambda path: FakePolicy())
    monkeypatch.setattr(module, "PrefixOutputCapture", FakeCapture)
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (object(), object()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache_v2.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--output-dir",
            str(tmp_path),
            "--max-episodes",
            "0",
            "--video-backend",
            "video_reader",
        ],
    )

    module.main()

    assert captured["video_backend"] == "video_reader"
    metadata = __import__("json").loads((tmp_path / "cache_metadata.json").read_text())
    assert metadata["format_version"] == 4
    assert metadata["build_complete"] is True
    assert metadata["splits"]["train"]["transitions"] == 0
    assert metadata["splits"]["val"]["transitions"] == 0
    assert (tmp_path / "chunk_transitions_train.pt").exists()
    assert (tmp_path / "chunk_transitions_val.pt").exists()
