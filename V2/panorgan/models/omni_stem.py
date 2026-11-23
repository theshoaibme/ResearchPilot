"""Omni-Modal Stem (2D + 2.5D Volumetric Slabs) with Physics-Informed FiLM Modulation."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on physical acquisition metadata.

    Modulates feature channels via: y = (1 + gamma(u)) * x + beta(u),
    where u encodes log voxel spacing (dx, dy, dz), dimensionality, contrast, and modality.
    Initialized to identity modulation (gamma=0, beta=0) for stable pretrained transfer.
    """

    def __init__(self, meta_dim: int, num_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, meta_vec: torch.Tensor) -> torch.Tensor:
        gb = self.net(meta_vec)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * features + beta


class OmniModalityStem(nn.Module):
    """Unified input stem handling 2D projections (CXR/MG/US) and 2.5D CT/MRI slabs.

    Eliminates depth-collapse encountered when passing D=1 images through 3D pooling layers
    by projecting 2D (3-ch) and 2.5D (K-slice) inputs via dedicated modality-appropriate
    convolutional stems into a harmonized 3-channel pseudo-RGB + FiLM-modulated feature space
    compatible with pretrained hierarchical vision backbones.
    """

    def __init__(
        self,
        slab_depth: int = 5,
        meta_dim: int = 10,
        stem_channels: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.slab_depth = slab_depth

        self.stem_2d = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
        )
        self.stem_25d = nn.Sequential(
            nn.Conv2d(slab_depth, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
        )
        self.physics_film = PhysicsFiLM(meta_dim=meta_dim, num_channels=stem_channels)
        self.to_backbone = nn.Conv2d(stem_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        meta_vec: torch.Tensor,
        is_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, slab_depth, H, W). For 2D samples, channels [0:3] carry data.
            meta_vec: Tensor of shape (B, meta_dim).
            is_3d: Boolean tensor of shape (B,) indicating 2.5D volumetric slabs vs 2D projections.
        Returns:
            Harmonized tensor of shape (B, 3, H, W) ready for pretrained vision backbone.
        """
        feat_2d = self.stem_2d(x[:, :3])
        feat_25d = self.stem_25d(x[:, : self.slab_depth])
        mask_3d = is_3d.to(dtype=feat_2d.dtype).view(-1, 1, 1, 1)
        fused = mask_3d * feat_25d + (1.0 - mask_3d) * feat_2d
        modulated = self.physics_film(fused, meta_vec)
        # Residual shortcut preserves raw intensity statistics for pretrained ImageNet backbones
        base_rgb = mask_3d * x[:, self.slab_depth // 2 : self.slab_depth // 2 + 1].repeat(1, 3, 1, 1) + (
            1.0 - mask_3d
        ) * x[:, :3]
        return base_rgb + 0.1 * self.to_backbone(modulated)
