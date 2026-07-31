from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer


def _transition(value: float, *, chunk_length: int = 2, action_dim: int = 4) -> ChunkTransition:
    return ChunkTransition(
        state_vec=torch.full((3,), value),
        exec_chunk=torch.full((chunk_length, action_dim), value),
        ref_chunk=torch.zeros(chunk_length, action_dim),
        reward_seq=torch.zeros(chunk_length),
        next_state_vec=torch.full((3,), value + 1),
        next_ref_chunk=torch.zeros(chunk_length, action_dim),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(chunk_length),
        episode_id=torch.tensor(int(value)),
    )


def _buffer(values: range) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=len(values))
    for value in values:
        replay.add(_transition(float(value)))
    return replay


def test_mixed_batch_has_fixed_offline_online_split():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(
        batch_size=10,
        offline_batch_fraction=0.4,
        use_stratified_sampling=False,
    )
    trainer.offline_buffer = _buffer(range(20))
    trainer.replay_buffer = _buffer(range(20, 40))

    batch, offline_n, online_n = trainer._sample_training_batch()

    assert offline_n == 4
    assert online_n == 6
    assert batch["state_vec"].shape[0] == 10


def test_offline_cache_loader_accepts_v2_dict_cache(tmp_path):
    transition = _transition(1.0)
    cache_dict = {
        key: getattr(transition, key)
        for key in ChunkTransition.__dataclass_fields__
        if isinstance(getattr(transition, key), torch.Tensor)
    }
    cache_path = tmp_path / "chunk_transitions_train.pt"
    torch.save([cache_dict], cache_path)
    (tmp_path / "cache_metadata.json").write_text(
        json.dumps(
            {
                "exec_chunk_source": "demonstrated_action",
                "rl_action_arms": "both",
            }
        )
    )

    replay = OnlineRLTrainer._load_offline_buffer(
        str(tmp_path),
        policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
    )

    assert replay is not None
    assert len(replay) == 1
    assert torch.equal(replay.buffer[0].exec_chunk, transition.exec_chunk)
