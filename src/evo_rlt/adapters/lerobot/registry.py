from __future__ import annotations

from pathlib import Path
from typing import Any

_REGISTERED = False


def register() -> None:
    """Register Evo-RLT policy/config factories with LeRobot at runtime.

    This keeps Evo-RLT out of LeRobot source files. Call once before using
    LeRobot factory helpers with policy.type in {rlt, rlt_token, rlt_ac}.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # 仿真机器人。``@RobotConfig.register_subclass("sim_bi_so_follower")`` 是在
    # **import 时**执行的,而 record 脚本的 ``--robot.type`` 选项列表是从注册表
    # 现场构建的 —— 没在这里 import 过,``--robot.type=sim_bi_so_follower`` 就会
    # 被 argparse 拒掉:"invalid choice"。
    #
    # 光在 ``evo_rlt.sim.__init__`` 里做惰性导出不够:那只在有人 getattr 的时候
    # 才触发,而 argparse 建选项列表时不会去 getattr。
    import evo_rlt.sim.sim_robot  # noqa: F401
    import lerobot.datasets.factory as dataset_factory
    import lerobot.policies.factory as factory

    from lerobot.configs.policies import PreTrainedConfig

    from evo_rlt.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    PreTrainedConfig._choice_registry["rlt"] = RLTPretrainedConfig
    PreTrainedConfig._choice_registry["rlt_ac"] = ChunkACPolicyConfig
    PreTrainedConfig._choice_registry["rlt_token"] = RLTokenPolicyConfig

    original_get_policy_class = factory.get_policy_class
    original_make_policy_config = factory.make_policy_config
    original_make_pre_post_processors = factory.make_pre_post_processors
    original_make_dataset = dataset_factory.make_dataset

    def get_policy_class(name: str):
        if name == "rlt_token":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy

            return RLTokenPolicy
        if name == "rlt_ac":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

            return ChunkACPolicy
        if name == "rlt":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy

            return RLTPretrainedPolicy
        return original_get_policy_class(name)

    def make_policy_config(policy_type: str, **kwargs: Any):
        if policy_type == "rlt_token":
            return RLTokenPolicyConfig(**kwargs)
        if policy_type == "rlt_ac":
            return ChunkACPolicyConfig(**kwargs)
        if policy_type == "rlt":
            return RLTPretrainedConfig(**kwargs)
        return original_make_policy_config(policy_type, **kwargs)

    def make_pre_post_processors(policy_cfg, *args: Any, **kwargs: Any):
        if isinstance(policy_cfg, RLTokenPolicyConfig):
            from evo_rlt.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors

            return make_rlt_token_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        if isinstance(policy_cfg, ChunkACPolicyConfig):
            from evo_rlt.adapters.lerobot.policies.processor_rlt_ac import make_rlt_ac_pre_post_processors

            return make_rlt_ac_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return original_make_pre_post_processors(policy_cfg, *args, **kwargs)

    def make_dataset(cfg):
        repo_id = getattr(cfg.dataset, "repo_id", None)
        cache_dir = Path(repo_id) if isinstance(repo_id, str) else None
        if cache_dir is not None and (cache_dir / "chunk_transitions_train.pt").exists():
            from evo_rlt.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset

            return ChunkTransitionDataset(cache_dir, split="train")
        return original_make_dataset(cfg)

    factory.get_policy_class = get_policy_class
    factory.make_policy_config = make_policy_config
    factory.make_pre_post_processors = make_pre_post_processors
    dataset_factory.make_dataset = make_dataset
    _REGISTERED = True
