from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

_DEFAULT_CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]


@dataclass
class ChunkACPolicyConfig(PreTrainedConfig):
    """Config for the chunk-level TD3+BC actor-critic policy on top of RL Token.

    Train via lerobot-train CLI with a ChunkTransitionDataset; deploy via
    make_policy + lerobot-record. forward(batch) returns a single scalar TD3+BC
    loss combining critic MSE + (every actor_update_interval steps) the actor
    Q-maximization + BC reg. Target critic is soft-updated with tau after every
    critic step. UTD ratio is achieved by setting outer --steps=outer*utd
    (each forward() == one critic update; actor update gated by counter).
    """

    # --- VLA + RL Token backbones (frozen, not serialized) ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"
    rl_token_pretrained_path: str = ""

    # --- RL Token arch (must match the loaded RLTokenPolicy ckpt) ---
    rl_token_dim: int = 2048
    rl_token_num_rl_tokens: int = 1
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Actor ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False
    # When True, actor predicts a residual delta added to the VLA reference
    # chunk (mu = ref + delta) with the final layer zero-initialized, so an
    # untrained actor starts out identical to the VLA reference. Used for
    # online RL on real hardware where a randomly-initialized actor would
    # otherwise be unsafe to run from step one.
    actor_residual_to_ref: bool = False
    # Which arm the actor may refine. "left" masks the residual for the right
    # half of each bimanual action vector, so those commands remain exactly the
    # frozen VLA reference during training, target computation, and rollout.
    actor_rl_arm: str = "both"
    # Optional safety bound on the RL actor's per-step deviation from the VLA
    # reference chunk while in the critical/RL phase: |action - ref| stays
    # under this value via project_action_delta's smooth ref+limit*tanh(...)
    # projection (see core/utils.py), applied identically in
    # RLTActionModifier.compute_chunk (deploy), actor_loss, and critic_loss's
    # target computation (training) -- NOT a plain clamp(mu-ref, -d, d),
    # which breaks the |action-ref|<=d bound whenever ref itself lies outside
    # [-1,1] (a real, common QUANTILES-normalization occurrence). None
    # disables the bound everywhere (existing pre-fix behavior: mu only ever
    # clamped to [-1,1], independent of ref).
    actor_action_clip_delta: float | None = None
    # Optional runtime slew-rate limit: caps how much the actor's *executed*
    # action may change from one physical timestep to the next (see
    # RLTActionModifier._apply_slew_rate_limit), independent of and in
    # addition to actor_action_clip_delta above -- the delta bound only
    # limits how far an action sits from its own reference, not how much it
    # can change from the immediately preceding frame, so adjacent frames
    # could otherwise swing from -actor_action_clip_delta to
    # +actor_action_clip_delta with nothing to stop them. None disables it
    # (existing behavior).
    actor_slew_rate_limit: float | None = None
    # Optional training-time complement to actor_slew_rate_limit: penalizes
    # squared differences between adjacent-timestep raw mu within a chunk
    # (see losses.actor_loss), discouraging the actor's own residual from
    # oscillating step-to-step in the first place rather than only clipping
    # the symptom at deploy time. 0.0 (default) is a no-op.
    actor_smoothness_weight: float = 0.0

    # --- Critic + target ---
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_activation: str = "relu"
    # RLPD (Ball et al., 2023): LayerNorm bounds Q by the output layer's
    # weight norm, mitigating catastrophic OOD value overestimation without
    # constraining exploration. target_q_clip below remains as a coarse
    # backstop, not the primary defense.
    critic_layer_norm: bool = True
    critic_residual: bool = False

    # --- TD3+BC hyperparams ---
    gamma: float = 0.99
    beta: float = 0.3
    tau: float = 0.005
    utd_ratio: int = 5
    actor_update_interval: int = 2
    target_q_clip: float = 100.0

    # --- RankQ self-supervised ranking loss (Choi & Xu, 2026) ---
    # Augments the critic TD loss with a pairwise softplus ranking term over
    # noisy/very-noisy/random/permuted variants of each executed action, so
    # Q-gradients w.r.t. action point toward higher-quality behavior instead
    # of just being clamped. Applied only where the sampled batch carries a
    # resolved "outcome" label (see losses.rankq_ranking_loss); batches
    # without one (e.g. offline lerobot-train pretraining) are unaffected.
    # alpha0/alpha1 = 1.0, noise_scale = 0.15 match the paper's defaults.
    # rankq_ranking_loss averages each branch over its own pair count before
    # applying alpha (see that function's docstring), so alpha=1.0 here is
    # already calibrated to a TD-loss-comparable scale -- it is NOT the raw,
    # un-normalized paper formula.
    rankq_alpha_success: float = 1.0
    rankq_alpha_failure: float = 1.0
    rankq_noise_scale: float = 0.15

    # TD3-style target policy smoothing: clipped noise added to the target
    # actor's action before evaluating target_critic on it, so the critic
    # can't fit an arbitrarily sharp function right at one exact action --
    # see losses.critic_loss's target_noise_std docstring for why this
    # matters specifically for real-robot jitter. On by default; pass
    # target_noise_std=0.0 to disable and get the exact pre-smoothing
    # behavior.
    target_noise_std: float = 0.1
    target_noise_clip: float = 0.3

    # --- Shapes ---
    chunk_length: int = 10
    action_dim: int = 12
    proprio_dim: int = 12

    # --- Deploy ---
    chunk_exec_steps: int = 25
    phase_mode: str = "always_rl"
    deterministic: bool = True

    # --- Observation mapping (for deploy preprocessor) ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))

    # --- Normalization MATCHES PI05 (deploy parity) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- pi05 proxy fields ---
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str | None = None

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt_ac"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got {self.phase_mode!r}"
            )
        if self.actor_rl_arm not in ("both", "left"):
            raise ValueError(
                f"actor_rl_arm must be 'both' or 'left', got {self.actor_rl_arm!r}"
            )
        if self.actor_rl_arm == "left" and self.action_dim % 2 != 0:
            raise ValueError("actor_rl_arm='left' requires an even action_dim")
        if self.actor_slew_rate_limit is not None and self.actor_slew_rate_limit < 0:
            raise ValueError("actor_slew_rate_limit must be >= 0 when set")
        if self.actor_smoothness_weight < 0:
            raise ValueError("actor_smoothness_weight must be >= 0")

    @property
    def type(self) -> str:
        return "rlt_ac"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.proprio_dim,)
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *self.image_resolution)
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None
