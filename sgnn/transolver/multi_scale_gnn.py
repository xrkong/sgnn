"""Transolver network with MLP encoder/decoder.

Architecture: MLP encoder -> stacked Transolver blocks -> MLP decoder.
"""

import torch
import torch.nn as nn

from sgnn.transolver.Transolver import Transolver_block


def build_mlp(
    input_size: int,
    hidden_layer_sizes: list[int],
    output_size: int | None = None,
    output_activation: type[nn.Module] = nn.Identity,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a simple MLP used for input/output projections."""
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size is not None:
        layer_sizes.append(output_size)

    nlayers = len(layer_sizes) - 1
    acts = [activation for _ in range(nlayers)]
    acts[-1] = output_activation

    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module(f"NN-{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module(f"Act-{i}", acts[i]())
    return mlp


class MultiScaleGNN(nn.Module):
    """MLP encoder + Transolver processor + MLP decoder."""

    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        block_act: str = "gelu",
        slice_num: int = 64,
        scale_hidden_dims: list[list[int]] | None = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.nmessage_passing_steps = nmessage_passing_steps

        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        # Keep this arg for backward-compatibility with newer configs.
        _ = scale_hidden_dims

        self.input_proj = nn.Sequential(
            build_mlp(
                nnode_in_features,
                [latent_dim for _ in range(nmlp_layers)],
                latent_dim,
            ),
            nn.LayerNorm(latent_dim),
        )

        # Keep head count compatible with hidden size.
        # candidate_heads = [8, 4, 2, 1]
        # n_head = next((h for h in candidate_heads if latent_dim % h == 0), 1)

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=num_heads,
                    hidden_dim=latent_dim,
                    dropout=dropout,
                    act=block_act,
                    mlp_ratio=mlp_ratio,
                    last_layer=False,
                    out_dim=nnode_out_features,
                    slice_num=slice_num,
                )
                for _ in range(nmessage_passing_steps)
            ]
        )

        self.output_head = build_mlp(
            latent_dim,
            [latent_dim for _ in range(nmlp_layers)],
            nnode_out_features,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass on node features.

        Supports either [N, C] for a single graph or [B, N, C] for a batch.
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 3:
            raise ValueError(f"Expected x with shape [N, C] or [B, N, C], got {tuple(x.shape)}")

        tokens = self.input_proj(x)

        for block in self.blocks:
            tokens = block(tokens)

        outputs = self.output_head(tokens)

        if squeeze_batch:
            outputs = outputs.squeeze(0)

        return outputs
