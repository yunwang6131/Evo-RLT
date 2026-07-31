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


def test_missing_demo_outcome_can_default_to_success():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    dataset = SimpleNamespace(meta=SimpleNamespace(episodes={}))
    assert module._get_episode_success(dataset, 0, default_outcome="success") is True


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
