"""Hierarchical Vision Backbone with LoRA, Sparse Top-2 MT-MoE, and SACR Head."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from panorgan.models.omni_stem import PhysicsFiLM


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for parameter-efficient foundation adaptation."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / float(max(rank, 1))
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling


def inject_lora_adapters(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Replace eligible projection nn.Linear layers in backbone blocks with LoRALinear."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= 64:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += inject_lora_adapters(child, rank=rank, alpha=alpha)
    return replaced


class _SpatialExpert(nn.Module):
    """Lightweight depthwise-separable ConvNeXt-style spatial expert block."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SparseMTMoELayer(nn.Module):
    """Sparse Top-K Modality-Task Mixture-of-Experts (MT-MoE).

    Routes spatial feature maps dynamically across specialized experts conditioned on
    pooled visual context + acquisition physics metadata, mitigating 2D/3D and Seg/Cls
    gradient interference while computing a Switch-style load-balancing auxiliary loss.
    """

    def __init__(
        self,
        channels: int,
        meta_dim: int = 10,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Sequential(
            nn.Linear(channels + meta_dim, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, num_experts),
        )
        self.experts = nn.ModuleList(
            [_SpatialExpert(channels, expansion=2, dropout=dropout) for _ in range(num_experts)]
        )
        self.norm = nn.GroupNorm(8, channels)

    def forward(
        self,
        feat: torch.Tensor,
        meta_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out: Residual-refined feature map (B, C, H, W).
            aux_loss: Scalar load-balancing loss encouraging balanced expert utilization.
            gate_probs: Routing distribution (B, num_experts) for interpretability logging.
        """
        pooled = feat.mean(dim=(-2, -1))
        router_in = torch.cat([pooled, meta_vec], dim=-1)
        logits = self.router(router_in)
        gate_probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute expert outputs and combine weighted Top-K per sample
        expert_stack = torch.stack([exp(feat) for exp in self.experts], dim=1)  # (B, E, C, H, W)
        b, _, c, h, w = expert_stack.shape
        gather_idx = topk_idx.view(b, self.top_k, 1, 1, 1).expand(b, self.top_k, c, h, w)
        selected = torch.gather(expert_stack, dim=1, index=gather_idx)
        weights_view = topk_weights.view(b, self.top_k, 1, 1, 1)
        routed = (selected * weights_view).sum(dim=1)

        # Switch Transformer load-balancing auxiliary loss
        dispatch_mask = torch.zeros_like(gate_probs).scatter_(1, topk_idx, 1.0 / self.top_k)
        f_i = dispatch_mask.mean(dim=0)
        p_i = gate_probs.mean(dim=0)
        aux_loss = float(self.num_experts) * torch.sum(f_i * p_i)

        out = self.norm(feat + routed)
        return out, aux_loss, gate_probs


class SACRHead(nn.Module):
    """Self-Supervised Anatomical Coordinate Regressor (SACR).

    Predicts normalized body-atlas coordinates (z_inf_sup, y_ap, x_lr) in [0, 1]^3 directly
    from deep visual appearance—solving MNAR partial-FOV scan truncation without circularly
    requiring a fixed anatomical landmark (e.g., carina) to be visible inside the scan.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )
        self.coord_embed = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
        )

    def forward(self, deepest_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.head(deepest_feat)
        anchor_bias = self.coord_embed(coords)[:, :, None, None]
        anchored_feat = deepest_feat + 0.1 * anchor_bias
        return coords, anchored_feat


class _LightweightFallbackBackbone(nn.Module):
    """Fast 4-stage hierarchical Conv backbone used in offline unit tests."""

    def __init__(self, dims: Tuple[int, int, int, int] = (48, 96, 192, 384)):
        super().__init__()
        self.dims = list(dims)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(8, dims[0]),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dims[1]),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dims[2]),
            nn.GELU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dims[3]),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class PanOrganBackbone(nn.Module):
    """Pretrained hierarchical encoder enhanced with PhysicsFiLM, LoRA, MT-MoE, and SACR."""

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        embed_dim: int = 384,
        meta_dim: int = 10,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        use_moe: bool = True,
        num_experts: int = 4,
        moe_top_k: int = 2,
        use_sacr: bool = True,
        grad_checkpointing: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_sacr = use_sacr
        self.embed_dim = embed_dim

        if backbone_name == "lightweight_test":
            self.encoder = _LightweightFallbackBackbone()
            stage_dims = self.encoder.dims
        else:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            stage_dims = list(self.encoder.feature_info.channels())
            if grad_checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
                self.encoder.set_grad_checkpointing(True)
            if use_lora:
                inject_lora_adapters(self.encoder, rank=lora_rank, alpha=lora_alpha)

        self.stage_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(8, embed_dim),
                    nn.GELU(),
                )
                for ch in stage_dims
            ]
        )
        self.stage_films = nn.ModuleList(
            [PhysicsFiLM(meta_dim=meta_dim, num_channels=embed_dim) for _ in stage_dims]
        )

        self.moe_stage3: Optional[SparseMTMoELayer] = (
            SparseMTMoELayer(embed_dim, meta_dim, num_experts, moe_top_k, dropout)
            if use_moe
            else None
        )
        self.moe_stage4: Optional[SparseMTMoELayer] = (
            SparseMTMoELayer(embed_dim, meta_dim, num_experts, moe_top_k, dropout)
            if use_moe
            else None
        )
        self.sacr_head: Optional[SACRHead] = SACRHead(embed_dim) if use_sacr else None

    def forward(
        self,
        x: torch.Tensor,
        meta_vec: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        raw_feats = self.encoder(x)
        # Ensure NCHW layout (some Swin backbones emit NHWC)
        feats: List[torch.Tensor] = []
        for idx, f in enumerate(raw_feats):
            if f.ndim == 4 and f.shape[-1] == self.stage_projs[idx][0].in_channels and f.shape[1] != self.stage_projs[idx][0].in_channels:
                f = f.permute(0, 3, 1, 2).contiguous()
            proj = self.stage_projs[idx](f)
            proj = self.stage_films[idx](proj, meta_vec)
            feats.append(proj)

        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        gate_probs: Optional[torch.Tensor] = None

        if self.use_moe and self.moe_stage3 is not None and self.moe_stage4 is not None:
            feats[2], aux3, _ = self.moe_stage3(feats[2], meta_vec)
            feats[3], aux4, gate_probs = self.moe_stage4(feats[3], meta_vec)
            aux_loss = 0.5 * (aux3 + aux4)

        pred_coords: Optional[torch.Tensor] = None
        if self.use_sacr and self.sacr_head is not None:
            pred_coords, feats[3] = self.sacr_head(feats[3])

        return feats, aux_loss, pred_coords, gate_probs
