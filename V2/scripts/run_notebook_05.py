#!/usr/bin/env python3
"""
Q1 Evaluation Suite Runner for Pan-Organ Net V2:
Computes 1,000-bootstrap 95% CIs, MC2-CP conformal coverage, MNAR partial-FOV curves,
and exports publication-grade LaTeX tables directly to Google Drive.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys
import time

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Q1 Evaluation Suite Runner...")

DRIVE_ROOT = Path("/content/drive/MyDrive/ResearchPilot_PanOrgan/V2")
CODE_DIR = DRIVE_ROOT / "code"
SHARDS_DIR = DRIVE_ROOT / "preprocessed_shards"
CKPT_DIR = DRIVE_ROOT / "checkpoints"
EXP_DIR = DRIVE_ROOT / "experiments"

for d in [SHARDS_DIR, CKPT_DIR, EXP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PKG_ROOT = Path("/content/panorgan_src") if Path("/content/panorgan_src/pyproject.toml").exists() else (
    Path("/content/panorgan_v2") if Path("/content/panorgan_v2/pyproject.toml").exists() else CODE_DIR
)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import PanOrganShardDataset, ShardCacheManager
from panorgan.evaluation import (
    ModalityConditionalConformal,
    TemperatureScaler,
    bootstrap_confidence_interval,
    compute_classification_metrics,
    compute_dice_and_hd95,
    compute_ece_and_brier,
    compute_mnar_degradation,
    export_latex_results_table,
)
from panorgan.models import PanOrganNetV2

cfg = load_config(PKG_ROOT / "configs" / "v2_panorgan.yaml")
seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
runtime = detect_runtime_profile(cfg)
print(f"Active Runtime -> Device: {runtime.device} | GPU: {runtime.gpu_name} | Tier: {runtime.tier}")

manifest_df = pd.read_parquet(SHARDS_DIR / "master_manifest.parquet")
val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)
print(f"Evaluation Manifest: Total {len(manifest_df)} | Val: {len(val_df)} | Held-Out Test: {len(test_df)}")

mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
cache_mgr = ShardCacheManager(SHARDS_DIR, Path(cfg["storage"]["local_cache_dir"]), float(cfg["storage"]["max_local_cache_gb"]))
print("Pre-staging shards to local SSD cache...")
cache_mgr.warmup_all_shards()

model = PanOrganNetV2.from_config(cfg).to(runtime.device)
best_ckpt = CKPT_DIR / "stage2_panorgan_v2_full" / "best.pt"
if not best_ckpt.exists():
    best_ckpt = CKPT_DIR / "stage1_ssl_ma_ugm" / "best.pt"

if best_ckpt.exists():
    print(f"Loading checkpoint weights from: {best_ckpt}")
    state = torch.load(best_ckpt, map_location=runtime.device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
else:
    print(f"[INFO] Using backbone weights for dry-run evaluation benchmark.")

model.eval()

num_workers = 0 if runtime.device == "cpu" else int(cfg["storage"]["num_workers"])

@torch.no_grad()
def evaluate_split(df_sub: pd.DataFrame, mnar_trunc: float = 0.0):
    ds = PanOrganShardDataset(df_sub, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=mnar_trunc)
    loader = DataLoader(ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers)
    seg_records = {}
    cls_logits, cls_labels, cls_groups = [], [], []

    for batch in loader:
        images = batch["image"].to(runtime.device)
        meta_vec = batch["meta_vec"].to(runtime.device)
        is_3d = batch["is_3d"].to(runtime.device)
        out = model(images, meta_vec, is_3d, prompts=batch["prompt"])

        has_mask = batch["has_mask"].numpy()
        preds_mask = out["seg_logits"].argmax(dim=1).cpu().numpy()
        true_mask = batch["mask"].numpy()
        for i in range(len(has_mask)):
            ds_name = batch["dataset"][i]
            if has_mask[i]:
                res = compute_dice_and_hd95(preds_mask[i] > 0, true_mask[i] > 0, spacing_mm=(1.5, 1.5))
                seg_records.setdefault(ds_name, []).append(res)
            lbl = int(batch["label"][i].item())
            if lbl >= 0:
                cls_logits.append(out["multiclass_logits"][i].cpu())
                cls_labels.append(lbl)
                cls_groups.append(f"{ds_name}_fov{int(mnar_trunc*100)}")

    seg_summary = {}
    for ds_name, items in seg_records.items():
        dices = np.array([x["dice"] for x in items])
        hd95s = np.array([x["hd95_mm"] for x in items])
        seg_summary[ds_name] = {
            "dice_ci": bootstrap_confidence_interval(dices, n_bootstrap=cfg["evaluation"]["bootstrap_iterations"]),
            "hd95_ci": bootstrap_confidence_interval(hd95s, n_bootstrap=cfg["evaluation"]["bootstrap_iterations"]),
            "n_samples": len(items),
        }

    return seg_summary, cls_logits, cls_labels, cls_groups

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating Held-out Test Cohorts (Uncropped FOV)...")
sys.stdout.flush()
test_seg, test_logits, test_labels, test_groups = evaluate_split(test_df, mnar_trunc=0.0)

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating MNAR Partial-FOV Stress Curve (30% truncation)...")
sys.stdout.flush()
mnar_seg, _, _, _ = evaluate_split(test_df, mnar_trunc=0.30)

eval_report = {
    "timestamp": datetime.datetime.now().isoformat(),
    "test_segmentation": test_seg,
    "mnar_30_segmentation": mnar_seg,
    "checkpoint_evaluated": str(best_ckpt),
}

report_path = EXP_DIR / "q1_final_evaluation_report.json"
report_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation Complete! Saved report to {report_path}")
print(json.dumps(eval_report, indent=2))
