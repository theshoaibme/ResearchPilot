#!/usr/bin/env python3
"""
Baselines and Ablations Runner for Pan-Organ Net V2.
Designed for headless execution via SSH on Colab runtime.
"""
from __future__ import annotations

import copy
import datetime
import json
import os
from pathlib import Path
import sys
import time

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Baselines & Ablations Runner...")

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

import pandas as pd
import torch
from torch.utils.data import DataLoader

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.engine import ColabResilientTrainer
from panorgan.models import PanOrganNetV2

cfg = load_config(PKG_ROOT / "configs" / "v2_panorgan.yaml")
seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
runtime = detect_runtime_profile(cfg)
print(f"Active Runtime -> Device: {runtime.device} | GPU: {runtime.gpu_name} | Tier: {runtime.tier} | Batch: {runtime.batch_size}x{runtime.grad_accum} | AMP: {runtime.amp_dtype}")

manifest_df = pd.read_parquet(SHARDS_DIR / "master_manifest.parquet")
train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)

mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
cache_mgr = ShardCacheManager(SHARDS_DIR, Path(cfg["storage"]["local_cache_dir"]), float(cfg["storage"]["max_local_cache_gb"]))
print("Pre-staging shards to local SSD cache...")
cache_mgr.warmup_all_shards()

train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=runtime.batch_size)
num_workers = 0 if runtime.device == "cpu" else int(cfg["storage"]["num_workers"])
pin_memory = runtime.device == "cuda"

train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

ABLATIONS = {
    "ablation_A1_random_mask": {"masking_mode": "random", "use_moe": True, "use_sacr": True},
    "ablation_A2_grad_sgm": {"masking_mode": "gradient_sgm", "use_moe": True, "use_sacr": True},
    "ablation_A3_no_moe": {"masking_mode": "ma_ugm", "use_moe": False, "use_sacr": True},
    "ablation_A4_no_sacr": {"masking_mode": "ma_ugm", "use_moe": True, "use_sacr": False},
}

summary = {}
for exp_name, overrides in ABLATIONS.items():
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Running {exp_name} ===")
    sys.stdout.flush()
    ab_cfg = copy.deepcopy(cfg)
    ab_cfg["model"].update(overrides)
    model = PanOrganNetV2.from_config(ab_cfg)
    trainer = ColabResilientTrainer(
        model=model,
        runtime=runtime,
        checkpoint_dir=CKPT_DIR / exp_name,
        lr=float(cfg["training"]["stage2_multitask"]["lr"]),
        patience=3,
    )
    epochs = 5 if runtime.device == "cpu" else 15
    trainer.fit(train_loader, val_loader, num_epochs=epochs, mode="multitask", resume=True)
    summary[exp_name] = {"best_val_loss": trainer.best_val_loss}

out_summary_path = EXP_DIR / "ablation_training_summary.json"
out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ablations Complete! Summary saved to {out_summary_path}")
print(json.dumps(summary, indent=2))
