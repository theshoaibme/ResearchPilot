"""Modality-Adaptive Uncertainty-Guided Masking (MA-UGM) for SSL pretraining and ablations."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAdaptiveUGM(nn.Module):
    """Patch-level masking generator supporting controlled Q1 ablations:

    - 'random': Standard uniform random masking (MAE baseline).
    - 'gradient_sgm': Raw intensity gradient inverse-saliency masking (V1 baseline).
    - 'ma_ugm': Modality-Adaptive Uncertainty-Guided Masking combining despeckled
      foreground saliency (warmup) with online EMA-teacher residual uncertainty.
    """

    def __init__(
        self,
        patch_size: int = 16,
        mask_ratio: float = 0.65,
        temperature: float = 0.25,
        mode: str = "ma_ugm",
    ):
        super().__init__()
        if mode not in ("random", "gradient_sgm", "ma_ugm"):
            raise ValueError(f"Unsupported masking mode: {mode}")
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.temperature = max(temperature, 1e-3)
        self.mode = mode

    def _compute_patch_scores(
        self,
        images: torch.Tensor,
        saliency: Optional[torch.Tensor],
        teacher_uncertainty: Optional[torch.Tensor],
        ema_blend: float,
    ) -> torch.Tensor:
        b, _, h, w = images.shape
        ph, pw = max(1, h // self.patch_size), max(1, w // self.patch_size)

        if self.mode == "random":
            return torch.rand((b, ph * pw), device=images.device)

        if self.mode == "gradient_sgm" or saliency is None:
            gray = images.mean(dim=1, keepdim=True)
            gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            sal_map = gx + gy
        else:
            sal_map = saliency.unsqueeze(1) if saliency.ndim == 3 else saliency

        patch_sal = F.adaptive_avg_pool2d(sal_map, output_size=(ph, pw)).flatten(1)
        patch_sal = patch_sal / (patch_sal.amax(dim=1, keepdim=True) + 1e-6)

        if self.mode == "ma_ugm" and teacher_uncertainty is not None and ema_blend > 0.0:
            unc_map = (
                teacher_uncertainty.unsqueeze(1)
                if teacher_uncertainty.ndim == 3
                else teacher_uncertainty
            )
            patch_unc = F.adaptive_avg_pool2d(unc_map, output_size=(ph, pw)).flatten(1)
            patch_unc = patch_unc / (patch_unc.amax(dim=1, keepdim=True) + 1e-6)
            combined = (1.0 - ema_blend) * patch_sal + ema_blend * patch_unc
        else:
            combined = patch_sal

        # Higher score -> higher anatomical/uncertainty importance -> lower probability of masking
        # so the encoder retains informative lesion/organ boundary tokens.
        logits = -combined / self.temperature
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward(
        self,
        images: torch.Tensor,
        saliency: Optional[torch.Tensor] = None,
        teacher_uncertainty: Optional[torch.Tensor] = None,
        ema_blend: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply patch masking to images.

        Returns:
            masked_images: Tensor (B, C, H, W) with masked patches zeroed.
            pixel_mask: Binary float Tensor (B, 1, H, W) where 1.0 = masked patch, 0.0 = visible.
        """
        b, _, h, w = images.shape
        ph, pw = max(1, h // self.patch_size), max(1, w // self.patch_size)
        num_patches = ph * pw
        num_mask = max(1, min(num_patches - 1, int(round(num_patches * self.mask_ratio))))

        weights = self._compute_patch_scores(images, saliency, teacher_uncertainty, ema_blend)
        if self.mode == "random":
            mask_indices = torch.topk(weights, k=num_mask, dim=-1).indices
        else:
            probs = torch.clamp(weights, min=1e-6)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            mask_indices = torch.multinomial(probs, num_samples=num_mask, replacement=False)

        patch_mask = torch.zeros((b, num_patches), device=images.device, dtype=images.dtype)
        patch_mask.scatter_(1, mask_indices, 1.0)
        pixel_mask = F.interpolate(
            patch_mask.view(b, 1, ph, pw),
            size=(h, w),
            mode="nearest",
        )
        masked_images = images * (1.0 - pixel_mask)
        return masked_images, pixel_mask
