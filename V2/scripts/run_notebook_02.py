#!/usr/bin/env python3
"""
Stage-1 SSL Pretraining Runner for Pan-Organ Net V2 (MA-UGM + SACR).
Designed for headless execution via SSH on Colab runtime.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import sys
import time

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Stage-1 SSL Runner...")

DRIVE_ROOT = Path("/content/drive/MyDrive/ResearchPilot_PanOrgan/V2")
CODE_DIR = DRIVE_ROOT / "code"
SHARDS_DIR = DRIVE_ROOT / "preprocessed_shards"
CKPT_DIR = DRIVE_ROOT / "checkpoints" / "stage1_ssl_ma_ugm"
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

manifest_path = SHARDS_DIR / "master_manifest.parquet"
if not manifest_path.exists():
    raise FileNotFoundError(f"Master manifest missing: {manifest_path}")

manifest_df = pd.read_parquet(manifest_path)
train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
print(f"Loaded Master Manifest: {len(manifest_df)} total samples | Train: {len(train_df)} | Val: {len(val_df)}")

mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
cache_mgr = ShardCacheManager(
    drive_shards_root=SHARDS_DIR,
    local_cache_dir=Path(cfg["storage"]["local_cache_dir"]),
    max_cache_gb=float(cfg["storage"]["max_local_cache_gb"]),
)
print("Pre-staging shards to local SSD cache...")
cache_mgr.warmup_all_shards()

train_ds = PanOrganShardDataset(
    train_df,
    cache_mgr,
    mod_map,
    target_hw=(256, 256),
    augment=True,
    mnar_truncation=0.25,
)
val_ds = PanOrganShardDataset(
    val_df,
    cache_mgr,
    mod_map,
    target_hw=(256, 256),
    augment=False,
    mnar_truncation=0.0,
)

sampler = BalancedModalityBatchSampler(
    train_df["dataset"].tolist(),
    batch_size=runtime.batch_size,
    temperature_alpha=cfg["training"]["stage2_multitask"]["sampling_temperature"],
)

num_workers = 0 if runtime.device == "cpu" else int(cfg["storage"]["num_workers"])
pin_memory = runtime.device == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_sampler=sampler,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
val_loader = DataLoader(
    val_ds,
    batch_size=runtime.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

model = PanOrganNetV2.from_config(cfg)
ssl_cfg = cfg["training"]["stage1_ssl"]

trainer = ColabResilientTrainer(
    model=model,
    runtime=runtime,
    checkpoint_dir=CKPT_DIR,
    lr=float(ssl_cfg["lr"]),
    weight_decay=float(ssl_cfg["weight_decay"]),
    sacr_loss_weight=float(ssl_cfg["sacr_loss_weight"]),
    moe_aux_weight=float(cfg["model"]["moe_aux_weight"]),
    checkpoint_every_steps=100,
)

epochs = int(ssl_cfg["epochs"])
warmup_epochs = int(ssl_cfg["warmup_epochs"])

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Stage-1 SSL Training ({epochs} epochs, warmup: {warmup_epochs})...")
sys.stdout.flush()

history = trainer.fit(
    train_loader,
    val_loader,
    num_epochs=epochs,
    mode="ssl",
    warmup_epochs=warmup_epochs,
    resume=True,
)

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage-1 SSL Completed Successfully!")
print(f"Best Val Loss: {trainer.best_val_loss:.6f}")
print(f"Checkpoint saved at: {CKPT_DIR / 'best.pt'}")
