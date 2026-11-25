#!/usr/bin/env python3
"""Controlled Ablation: Standard 2D ConvNeXt-Small (no OmniModalityStem/LoRA/MoE)
with reduced LR (3e-5) and stronger regularization (WD=0.08, DropPath=0.25, Dropout=0.4, LabelSmoothing=0.05)
on the 5 weak cohorts: cmmd_mg, cbis_ddsm, kidney_ct_abn, tcga_kirc_ct, ultrasound_nerve.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_4"))
os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "V2"))
sys.path.insert(0, str(_REPO_ROOT / "V2" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V3" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V4" / "scripts"))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from PanOrgan_4_train import log_msg
from train_v3_pipeline import atomic_torch_save, atomic_write_json

WEAK_COHORTS = ["cmmd_mg", "cbis_ddsm", "kidney_ct_abn", "tcga_kirc_ct", "ultrasound_nerve"]


class StandardConvNeXtAblation(nn.Module):
    def __init__(self, cohorts: List[str], drop_path_rate: float = 0.25, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k_384",
            pretrained=True,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features  # 768 for convnext_small
        self.heads = nn.ModuleDict({
            c: nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.Linear(256, 2),
            )
            for c in cohorts
        })

    def extract_rgb(self, x: torch.Tensor, is_3d: torch.Tensor) -> torch.Tensor:
        # 2D: x[:, :3]; 2.5D CT/MRI: 3 central slices x[:, 1:4]
        mask_3d = is_3d.view(-1, 1, 1, 1).bool()
        return torch.where(mask_3d, x[:, 1:4], x[:, :3])

    def forward(self, x: torch.Tensor, is_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = self.extract_rgb(x, is_3d)
        feat = self.backbone(rgb)
        return {c: head(feat) for c, head in self.heads.items()}


@torch.no_grad()
def evaluate_split(model: StandardConvNeXtAblation, loader: DataLoader, device: torch.device, tta: bool = True) -> Dict[str, Any]:
    model.eval()
    by_c: Dict[str, Dict[str, list]] = {c: {"logits": [], "labels": []} for c in WEAK_COHORTS}
    total_loss = 0.0
    n_loss = 0
    for batch in loader:
        images = batch["image"].to(device)
        is_3d = batch["is_3d"].to(device)
        out1 = model(images, is_3d)
        if tta:
            out2 = model(torch.flip(images, dims=[-1]), is_3d)
            logits_map = {c: 0.5 * (out1[c] + out2[c]) for c in WEAK_COHORTS}
        else:
            logits_map = out1

        for i, ds in enumerate(batch["dataset"]):
            lbl = int(batch["label"][i].item())
            if lbl < 0 or lbl >= 2 or ds not in logits_map:
                continue
            lg = logits_map[ds][i : i + 1]
            loss = F.cross_entropy(lg, torch.tensor([lbl], device=device))
            total_loss += float(loss.item())
            n_loss += 1
            by_c[ds]["logits"].append(lg[0].cpu())
            by_c[ds]["labels"].append(lbl)

    per_cohort: Dict[str, Dict[str, float]] = {}
    all_y, all_pred, f1s, aucs = [], [], [], []
    for ds in WEAK_COHORTS:
        if not by_c[ds]["logits"]:
            continue
        lg = torch.stack(by_c[ds]["logits"])
        y = np.array(by_c[ds]["labels"])
        p = F.softmax(lg, dim=-1).numpy()
        pred = p.argmax(axis=-1)
        acc = float(accuracy_score(y, pred))
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        auroc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) > 1 else 0.5
        per_cohort[ds] = {"n": int(len(y)), "accuracy": acc, "macro_f1": mf1, "auroc": auroc}
        all_y.extend(y.tolist())
        all_pred.extend(pred.tolist())
        f1s.append(mf1)
        aucs.append(auroc)

    return {
        "loss": total_loss / max(1, n_loss),
        "weighted_acc": float(accuracy_score(all_y, all_pred)) if all_y else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_auroc": float(np.mean(aucs)) if aucs else 0.5,
        "per_cohort": per_cohort,
    }


def main() -> None:
    seed_everything(42)
    cfg = load_config(str(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml"))
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    manifest_path = _V3_SSD / "preprocessed_shards" / "master_manifest_v4.parquet"
    df = pd.read_parquet(manifest_path)
    df = df[df["dataset"].isin(WEAK_COHORTS) & (df["label"] >= 0) & (df["label"] < 2)].reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    shards_dir = _V3_SSD / "preprocessed_shards"
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True)
    val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    test_ds = PanOrganShardDataset(test_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

    sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=16, temperature_alpha=0.25)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=0)

    # Class weights per cohort
    cw_map: Dict[str, torch.Tensor] = {}
    for ds in WEAK_COHORTS:
        sub = train_df[train_df["dataset"] == ds]["label"].values
        counts = np.bincount(sub, minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (2.0 * counts)
        w = w / w.mean()
        cw_map[ds] = torch.tensor(w, dtype=torch.float32, device=device)

    model = StandardConvNeXtAblation(WEAK_COHORTS, drop_path_rate=0.25, dropout=0.4).to(device)
    epochs = 6
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 3e-5, "weight_decay": 0.08},
            {"params": model.heads.parameters(), "lr": 1.5e-4, "weight_decay": 0.05},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)

    out_dir = _SSD_ROOT / "experiments" / "ablation_convnext_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "ablation.log"
    log_msg(f"Starting Standard 2D ConvNeXt-Small Ablation on {device} | train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", log_file)

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            is_3d = batch["is_3d"].to(device)
            labels = batch["label"].to(device)
            datasets = batch["dataset"]

            optimizer.zero_grad()
            logits_map = model(images, is_3d)
            losses = []
            for i, ds in enumerate(datasets):
                lbl = int(labels[i].item())
                if lbl < 0 or lbl >= 2 or ds not in logits_map:
                    continue
                ce = F.cross_entropy(
                    logits_map[ds][i : i + 1],
                    labels[i : i + 1],
                    weight=cw_map[ds],
                    label_smoothing=0.05,
                )
                losses.append(ce)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += float(loss.item())
            tr_n += 1

        val_res = evaluate_split(model, val_loader, device, tta=False)
        elapsed = time.time() - t0
        tr_loss_avg = tr_loss / max(1, tr_n)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss_avg,
            "val_loss": val_res["loss"],
            "val_weighted_acc": val_res["weighted_acc"],
            "val_mean_macro_f1": val_res["mean_macro_f1"],
            "val_mean_auroc": val_res["mean_auroc"],
        })
        log_msg(
            f"[ConvNeXt-Small Ep {epoch}/{epochs} ({elapsed:.0f}s)] "
            f"tr_loss={tr_loss_avg:.4f} | val_loss={val_res['loss']:.4f} | "
            f"val_acc={val_res['weighted_acc']:.4f} | val_F1={val_res['mean_macro_f1']:.4f} | val_AUROC={val_res['mean_auroc']:.4f}",
            log_file,
        )
        if val_res["mean_macro_f1"] > best_f1:
            best_f1 = val_res["mean_macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate_split(model, test_loader, device, tta=True)
    log_msg(f"=== TEST RESULTS (Standard ConvNeXt-Small): WeightedAcc={test_res['weighted_acc']:.4f} | MeanF1={test_res['mean_macro_f1']:.4f} | MeanAUROC={test_res['mean_auroc']:.4f} ===", log_file)
    for ds, m in test_res["per_cohort"].items():
        log_msg(f"  {ds:18s}: n={m['n']:3d} | Acc={m['accuracy']*100:.2f}% | F1={m['macro_f1']:.4f} | AUROC={m['auroc']:.4f}", log_file)

    atomic_write_json(out_dir / "ablation_results.json", {"history": history, "test": test_res})
    atomic_torch_save({"model_state": best_state}, out_dir / "convnext_small_ablation_best.pt")


if __name__ == "__main__":
    main()
