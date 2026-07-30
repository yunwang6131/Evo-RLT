from __future__ import annotations

import torch
import torch.nn as nn

from evo_rlt.core.utils import build_mlp, _get_activation


class ResidualMLP(nn.Module):
    """MLP with residual connections between hidden layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block_layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(_get_activation(activation))
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


class ChunkActor(nn.Module):
    """Actor that predicts an action chunk conditioned on RL state and VLA reference.

    Uses binary reference dropout (per batch element) during training to avoid
    over-reliance on the VLA reference chunk.
    """

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
        ref_dropout_p: float = 0.5,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
        residual_to_ref: bool = False,
        action_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if residual:
            self.net = ResidualMLP(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        else:
            self.net = build_mlp(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p
        self.residual_to_ref = residual_to_ref
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32).flatten()
            if action_mask.numel() != chunk_dim:
                raise ValueError(
                    f"action_mask must have {chunk_dim} entries, got {action_mask.numel()}"
                )
        # Derived from policy config, so it need not be stored in state_dict.
        # This keeps checkpoints made before action masking load-compatible.
        self.register_buffer("action_mask", action_mask, persistent=False)
        if residual_to_ref:
            # Zero-init the final layer so mu == ref_chunk exactly at init
            # (delta == 0), giving a safe starting policy for online RL on
            # real hardware: the untrained actor is a no-op over the VLA
            # reference until gradient updates move it away from zero.
            out_layer = self.net.output_proj if residual else self.net[-1]
            nn.init.zeros_(out_layer.weight)
            nn.init.zeros_(out_layer.bias)

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, std)."""
        net_ref = ref_chunk_flat
        if training:
            mask = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device) > self.ref_dropout_p
            ).float()
            net_ref = ref_chunk_flat * mask
        x = torch.cat([state_vec, net_ref], dim=-1)
        delta = self.net(x)
        mu = ref_chunk_flat + delta if self.residual_to_ref else delta
        if self.action_mask is not None:
            # Masking is expressed relative to the VLA reference even for a
            # non-residual actor: masked dimensions must remain VLA commands,
            # never zero actions. For residual actors this simplifies to
            # ref + mask * delta.
            mask = self.action_mask.to(dtype=mu.dtype)
            mu = ref_chunk_flat + mask * (mu - ref_chunk_flat)
        std = torch.full_like(mu, self.fixed_std)
        if self.action_mask is not None:
            # Sampling must not reintroduce noise on VLA-only dimensions.
            std = std * self.action_mask.to(dtype=std.dtype)
        return mu, std

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with Gaussian noise. Returns (action, mu)."""
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        return mu + std * torch.randn_like(std), mu
