"""Unified Pan-Organ Net V2 Model & Identical-Split Comparison Baselines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from panorgan.models.decoders import (
    ClinicalPathologyHead,
    OntologyPromptEncoder,
    UniversalMaskDecoder,
)
from panorgan.models.ma_ugm import ModalityAdaptiveUGM
from panorgan.models.mt_moe_backbone import PanOrganBackbone
from panorgan.models.omni_stem import OmniModalityStem


class PanOrganNetV2(nn.Module):
    """Unified 2D + 2.5D Multi-Modality Clinical Foundation Model.

    Combines:
      1. OmniModalityStem + PhysicsFiLM (CT/MRI/CXR/US/MG harmonization)
      2. ModalityAdaptiveUGM (Stage-1 SSL hard-region masking)
      3. Pretrained Hierarchical Backbone + LoRA + Sparse Top-2 MT-MoE
      4. Self-Supervised Anatomical Coordinate Regressor (SACR for MNAR FOV robustness)
      5. Prompt-Conditioned Universal Mask Decoder + Clinical Pathology Head
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        embed_dim: int = 384,
        text_dim: int = 512,
        meta_dim: int = 10,
        slab_depth: int = 5,
        max_seg_classes: int = 16,
        num_multilabel: int = 14,
        max_multiclass: int = 4,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        use_moe: bool = True,
        num_experts: int = 4,
        moe_top_k: int = 2,
        use_sacr: bool = True,
        masking_mode: str = "ma_ugm",
        mask_ratio: float = 0.65,
        mask_temperature: float = 0.25,
        grad_checkpointing: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.omni_stem = OmniModalityStem(
            slab_depth=slab_depth,
            meta_dim=meta_dim,
            stem_channels=64,
            out_channels=3,
        )
        self.ugm = ModalityAdaptiveUGM(
            patch_size=16,
            mask_ratio=mask_ratio,
            temperature=mask_temperature,
            mode=masking_mode,
        )
        self.backbone = PanOrganBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
            meta_dim=meta_dim,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            use_sacr=use_sacr,
            grad_checkpointing=grad_checkpointing,
            dropout=dropout,
        )
        self.prompt_encoder = OntologyPromptEncoder(text_dim=text_dim, out_dim=embed_dim)
        self.mask_decoder = UniversalMaskDecoder(embed_dim=embed_dim, max_seg_classes=max_seg_classes)
        self.pathology_head = ClinicalPathologyHead(
            embed_dim=embed_dim,
            num_multilabel=num_multilabel,
            max_multiclass=max_multiclass,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PanOrganNetV2":
        m = cfg["model"]
        return cls(
            backbone_name=str(m["backbone_name"]),
            pretrained=bool(m.get("pretrained", True)),
            embed_dim=int(m.get("embed_dim", 384)),
            text_dim=int(m.get("text_dim", 512)),
            meta_dim=int(m.get("meta_input_dim", 10)),
            slab_depth=int(m.get("slab_depth", 5)),
            use_lora=bool(m.get("use_lora", True)),
            lora_rank=int(m.get("lora_rank", 8)),
            lora_alpha=float(m.get("lora_alpha", 16.0)),
            use_moe=bool(m.get("use_moe", True)),
            num_experts=int(m.get("num_experts", 4)),
            moe_top_k=int(m.get("moe_top_k", 2)),
            use_sacr=bool(m.get("use_sacr", True)),
            masking_mode=str(m.get("masking_mode", "ma_ugm")),
            mask_ratio=float(m.get("mask_ratio", 0.65)),
            mask_temperature=float(m.get("mask_temperature", 0.25)),
            grad_checkpointing=bool(m.get("grad_checkpointing", True)),
            dropout=float(m.get("dropout", 0.2)),
        )

    def forward_ssl(
        self,
        images: torch.Tensor,
        meta_vec: torch.Tensor,
        is_3d: torch.Tensor,
        saliency: Optional[torch.Tensor] = None,
        teacher_uncertainty: Optional[torch.Tensor] = None,
        ema_blend: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Stage-1 Self-Supervised Pretraining (MA-UGM reconstruction + SACR coordinates)."""
        harmonized = self.omni_stem(images, meta_vec, is_3d)
        masked_x, pixel_mask = self.ugm(
            harmonized,
            saliency=saliency,
            teacher_uncertainty=teacher_uncertainty,
            ema_blend=ema_blend,
        )
        feats, moe_aux_loss, pred_coords, gate_probs = self.backbone(masked_x, meta_vec)
        recon = self.mask_decoder.reconstruct_ssl(feats, out_hw=(images.shape[-2], images.shape[-1]))
        # Target normalized in [0, 1]
        target = harmonized.detach().clamp(0.0, 1.0)
        recon_err_map = (recon - target).abs().mean(dim=1)
        masked_mse = (((recon - target) ** 2) * pixel_mask).sum() / (pixel_mask.sum() * 3.0 + 1e-6)
        return {
            "recon": recon,
            "recon_loss": masked_mse,
            "recon_err_map": recon_err_map.detach(),
            "pixel_mask": pixel_mask,
            "moe_aux_loss": moe_aux_loss,
            "pred_coords": pred_coords,
            "gate_probs": gate_probs,
        }

    def forward(
        self,
        images: torch.Tensor,
        meta_vec: torch.Tensor,
        is_3d: torch.Tensor,
        prompts: Sequence[str],
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """Stage-2 Supervised Multi-Task & Inference Forward Pass."""
        harmonized = self.omni_stem(images, meta_vec, is_3d)
        feats, moe_aux_loss, pred_coords, gate_probs = self.backbone(harmonized, meta_vec)
        prompt_emb = self.prompt_encoder(prompts, device=images.device)

        seg_logits = self.mask_decoder(
            feats,
            prompt_emb,
            out_hw=(images.shape[-2], images.shape[-1]),
        )
        cls_out = self.pathology_head(feats[-1], prompt_emb)

        out: Dict[str, Any] = {
            "seg_logits": seg_logits,
            "multilabel_logits": cls_out["multilabel_logits"],
            "multiclass_logits": cls_out["multiclass_logits"],
            "embedding": cls_out["embedding"],
            "moe_aux_loss": moe_aux_loss,
            "pred_coords": pred_coords,
            "gate_probs": gate_probs,
        }
        if return_features:
            out["features"] = feats
        return out


class UNet25DBaseline(nn.Module):
    """Clean single-task 2D/2.5D U-Net baseline for identical-split segmentation benchmarking."""

    def __init__(self, in_channels: int = 5, num_classes: int = 16, base_c: int = 32):
        super().__init__()

        def _conv(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.e1 = _conv(in_channels, base_c)
        self.e2 = _conv(base_c, base_c * 2)
        self.e3 = _conv(base_c * 2, base_c * 4)
        self.b = _conv(base_c * 4, base_c * 8)
        self.pool = nn.MaxPool2d(2)
        self.d3 = _conv(base_c * 8 + base_c * 4, base_c * 4)
        self.d2 = _conv(base_c * 4 + base_c * 2, base_c * 2)
        self.d1 = _conv(base_c * 2 + base_c, base_c)
        self.head = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        bn = self.b(self.pool(c3))
        u3 = self.d3(torch.cat([F.interpolate(bn, size=c3.shape[-2:], mode="bilinear", align_corners=False), c3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=c2.shape[-2:], mode="bilinear", align_corners=False), c2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=c1.shape[-2:], mode="bilinear", align_corners=False), c1], dim=1))
        return self.head(u1)


class BaselineModelFactory:
    """Constructs single-task clinical baselines evaluated on identical patient splits."""

    @staticmethod
    def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
        mapping = {
            "densenet121": "densenet121",
            "convnext_small": "convnext_small.fb_in22k_ft_in1k_384",
            "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
        }
        key = name.lower()
        if key not in mapping:
            raise ValueError(f"Unknown baseline classifier '{name}'. Choose from {list(mapping)}")
        return timm.create_model(mapping[key], pretrained=pretrained, num_classes=num_classes)

    @staticmethod
    def build_segmenter(in_channels: int = 5, num_classes: int = 16) -> nn.Module:
        return UNet25DBaseline(in_channels=in_channels, num_classes=num_classes)
