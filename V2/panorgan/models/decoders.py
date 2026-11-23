"""Prompt-conditioned Universal Mask Decoder, Clinical Pathology Head, and Clinical Losses."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class OntologyPromptEncoder(nn.Module):
    """Encodes SNOMED/RadLex-standardized clinical prompts into dense query embeddings.

    Supports precomputed BiomedCLIP/PubMedBERT embeddings loaded from cache during Colab
    training, with a deterministic character-ngram projection fallback for offline tests.
    """

    def __init__(self, text_dim: int = 512, out_dim: int = 384):
        super().__init__()
        self.text_dim = text_dim
        self.out_dim = out_dim
        self._cache: Dict[str, torch.Tensor] = {}
        self.proj = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def register_precomputed(self, prompt_to_embedding: Dict[str, torch.Tensor]) -> None:
        for prompt, emb in prompt_to_embedding.items():
            self._cache[prompt] = emb.detach().float().cpu()

    def _deterministic_embed(self, prompt: str) -> torch.Tensor:
        if prompt in self._cache:
            return self._cache[prompt]
        vec = torch.zeros(self.text_dim, dtype=torch.float32)
        tokens = prompt.lower().split() or ["medical", "scan"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "big") % self.text_dim
                sign = 1.0 if (digest[i] % 2 == 0) else -1.0
                vec[idx] += sign
        norm = vec.norm(p=2).clamp(min=1e-6)
        vec = vec / norm
        self._cache[prompt] = vec
        return vec

    def forward(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        raw = torch.stack([self._deterministic_embed(p) for p in prompts], dim=0).to(device)
        return self.proj(raw)


class UniversalMaskDecoder(nn.Module):
    """Multi-scale FPN pixel decoder conditioned on ontology prompt queries.

    Produces dense organ/lesion segmentation logits at input spatial resolution.
    """

    def __init__(self, embed_dim: int = 384, max_seg_classes: int = 16):
        super().__init__()
        self.max_seg_classes = max_seg_classes
        self.lat3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.lat2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.lat1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.norms = nn.ModuleList([nn.GroupNorm(8, embed_dim) for _ in range(3)])

        # Cross-attention conditioning between visual FPN features and text prompt query
        self.prompt_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, max_seg_classes, kernel_size=1),
        )
        # Lightweight reconstruction head for Stage-1 SSL (MA-UGM)
        self.ssl_recon_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def _fuse_fpn(self, feats: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        p3 = F.gelu(self.norms[0](self.lat3(f3 + F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False))))
        p2 = F.gelu(self.norms[1](self.lat2(f2 + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False))))
        p1 = F.gelu(self.norms[2](self.lat1(f1 + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False))))
        return p1

    def forward(
        self,
        feats: List[torch.Tensor],
        prompt_emb: torch.Tensor,
        out_hw: tuple[int, int],
    ) -> torch.Tensor:
        p1 = self._fuse_fpn(feats)
        gb = self.prompt_gate(prompt_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        conditioned = p1 * (1.0 + 0.1 * gamma[:, :, None, None]) + 0.1 * beta[:, :, None, None]
        logits = self.refine(conditioned)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)

    def reconstruct_ssl(self, feats: List[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        p1 = self._fuse_fpn(feats)
        recon = self.ssl_recon_head(p1)
        return F.interpolate(recon, size=out_hw, mode="bilinear", align_corners=False)


class ClinicalPathologyHead(nn.Module):
    """Prompt-conditioned clinical classification heads (multi-label thoracic + multi-class lesion)."""

    def __init__(self, embed_dim: int = 384, num_multilabel: int = 14, max_multiclass: int = 4, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.multilabel_head = nn.Linear(embed_dim, num_multilabel)
        self.multiclass_head = nn.Linear(embed_dim, max_multiclass)

    def forward(self, deepest_feat: torch.Tensor, prompt_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(deepest_feat).flatten(1)
        fused = self.fusion(torch.cat([pooled, prompt_emb], dim=-1))
        return {
            "multilabel_logits": self.multilabel_head(fused),
            "multiclass_logits": self.multiclass_head(fused),
            "embedding": fused,
        }


class DiceFocalBoundaryLoss(nn.Module):
    """Composite segmentation loss combining Multi-Class Soft Dice, Focal CE, and Boundary Sobel Loss.

    Prevents dominant large organs/background from overwhelming small tumors and fine margins.
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0, boundary_weight: float = 0.3, gamma: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_active_classes: Optional[int] = None) -> torch.Tensor:
        if num_active_classes is not None and num_active_classes < logits.shape[1]:
            logits = logits[:, :num_active_classes]
        num_classes = logits.shape[1]
        targets = targets.clamp(0, num_classes - 1)

        # 1. Focal Cross-Entropy
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (((1.0 - pt) ** self.gamma) * ce).mean()

        # 2. Multi-Class Soft Dice (excluding background channel 0 when foreground classes exist)
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        start_c = 1 if num_classes > 1 else 0
        p_fg = probs[:, start_c:]
        t_fg = one_hot[:, start_c:]
        inter = (p_fg * t_fg).sum(dim=(0, 2, 3))
        denom = p_fg.sum(dim=(0, 2, 3)) + t_fg.sum(dim=(0, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()

        # 3. Spatial Boundary Gradient Alignment Loss
        px = torch.abs(p_fg[:, :, :, 1:] - p_fg[:, :, :, :-1])
        tx = torch.abs(t_fg[:, :, :, 1:] - t_fg[:, :, :, :-1])
        py = torch.abs(p_fg[:, :, 1:, :] - p_fg[:, :, :-1, :])
        ty = torch.abs(t_fg[:, :, 1:, :] - t_fg[:, :, :-1, :])
        boundary_loss = 0.5 * (F.l1_loss(px, tx) + F.l1_loss(py, ty))

        return self.dice_weight * dice_loss + self.focal_weight * focal + self.boundary_weight * boundary_loss


class AsymmetricFocalLoss(nn.Module):
    """Clinical cost-sensitive multi-label Asymmetric Loss penalizing missed pathologies (FN)."""

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 3.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets >= 0
        if not valid.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        logits = logits[valid]
        targets = targets[valid].float()

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=1e-7))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-7))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, one_sided_gamma)

        return -loss.mean()
