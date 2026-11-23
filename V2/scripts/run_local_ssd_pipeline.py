#!/usr/bin/env python3
"""Power-outage (loadshedding) resilient local SSD downloader, trainer, and evaluator.

Features:
1. Atomic Drive-to-SSD download sync (.tmp -> fsync -> replace) with exact per-file skip/resume.
2. Exact mid-epoch batch-step checkpointing (every 50 steps) with fsync + atomic replace.
3. Persistent pipeline_state.json tracking completed stages so restarts resume at the exact stage & step.
4. Native Apple Silicon M4 GPU (MPS) acceleration via detect_runtime_profile.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Route all model/dataset caches to external SSD before importing torch/timm/hf
_DEFAULT_SSD = os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_V2")
os.environ.setdefault("HF_HOME", str(Path(_DEFAULT_SSD) / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(Path(_DEFAULT_SSD) / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(Path(_DEFAULT_SSD) / "model_cache" / "timm"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_V2 = Path(__file__).resolve().parent.parent
if str(REPO_V2) not in sys.path:
    sys.path.insert(0, str(REPO_V2))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.engine import ColabResilientTrainer
from panorgan.evaluation import (
    bootstrap_confidence_interval,
    compute_dice_and_hd95,
)
from panorgan.models import PanOrganNetV2


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_msg(msg: str, log_file: Path) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _gdrive_json(args: List[str]) -> Any:
    cmd = ["gdrive", "readonly"] + args + ["--json"]
    last_err = None
    for attempt in range(1, 4):
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if res.returncode == 0:
            return json.loads(res.stdout)
        last_err = f"rc={res.returncode}, stdout={res.stdout}, stderr={res.stderr}"
        time.sleep(2 * attempt)
    raise RuntimeError(f"gdrive command failed: {' '.join(cmd)} -> {last_err}")


def _list_drive_children(folder_id: str) -> List[Dict[str, Any]]:
    data = _gdrive_json(["ls", folder_id])
    if isinstance(data, list):
        return data
    return data.get("files", [])


def sync_drive_folder_resilient(
    folder_id: str,
    local_dir: Path,
    log_file: Path,
) -> None:
    """Recursively download a Drive folder to local SSD with atomic skip/resume."""
    local_dir.mkdir(parents=True, exist_ok=True)
    items = _list_drive_children(folder_id)
    for item in items:
        name = item["name"]
        item_id = item["id"]
        mime = item.get("mimeType", "")
        if mime == "application/vnd.google-apps.folder":
            sync_drive_folder_resilient(item_id, local_dir / name, log_file)
            continue

        expected_size = int(item.get("size", 0))
        dest = local_dir / name
        if dest.exists() and (expected_size == 0 or dest.stat().st_size == expected_size):
            log_msg(f"[SYNC SKIP] Already complete: {dest} ({dest.stat().st_size} bytes)", log_file)
            continue

        tmp = local_dir / f".{name}.part"
        if tmp.exists():
            tmp.unlink()
        log_msg(f"[SYNC DOWN] Downloading {name} ({expected_size / (1024**2):.1f} MB) -> {dest}", log_file)
        for attempt in range(1, 4):
            res = subprocess.run(
                ["gdrive", "readonly", "download", item_id, "--out", str(tmp)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                break
            log_msg(f"[SYNC RETRY {attempt}/3] {name}: {res.stderr.strip()}", log_file)
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise RuntimeError(f"Download failed for {name}: {res.stderr}")
            time.sleep(3 * attempt)
        if expected_size > 0 and tmp.stat().st_size != expected_size:
            raise RuntimeError(f"Size mismatch for {name}: got {tmp.stat().st_size}, expected {expected_size}")
        os.replace(tmp, dest)
        log_msg(f"[SYNC DONE] Saved {dest}", log_file)


def sync_from_drive(ssd_root: Path, v2_drive_id: str, log_file: Path) -> None:
    """Sync preprocessed_shards and any existing stage1_ssl checkpoint from Drive to SSD."""
    log_msg(f"=== Checking Google Drive sync -> {ssd_root} ===", log_file)
    children = {c["name"]: c["id"] for c in _list_drive_children(v2_drive_id)}
    if "preprocessed_shards" not in children:
        raise RuntimeError("preprocessed_shards folder not found in Drive V2 root.")
    sync_drive_folder_resilient(
        children["preprocessed_shards"],
        ssd_root / "preprocessed_shards",
        log_file,
    )
    if "checkpoints" in children:
        ckpt_children = {c["name"]: c["id"] for c in _list_drive_children(children["checkpoints"])}
        if "stage1_ssl_ma_ugm" in ckpt_children:
            sync_drive_folder_resilient(
                ckpt_children["stage1_ssl_ma_ugm"],
                ssd_root / "checkpoints" / "stage1_ssl_ma_ugm",
                log_file,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Power-outage resilient local SSD pipeline")
    parser.add_argument("--ssd-root", type=Path, default=Path("/Volumes/Shoaib/PanOrgan_V2"))
    parser.add_argument("--v2-drive-id", type=str, default="1kPGZQpK6EyoFLQxG9dBuRq_F39y-Nuws")
    parser.add_argument("--skip-sync", action="store_true", help="Skip Drive download sync if shards are already local")
    parser.add_argument("--sync-only", action="store_true", help="Only sync Drive files to SSD and exit")
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    args = parser.parse_args()

    ssd_root: Path = args.ssd_root
    os.environ["PANORGAN_DRIVE_ROOT"] = str(ssd_root)
    os.environ["PANORGAN_CACHE_DIR"] = str(ssd_root / "local_cache")

    shards_dir = ssd_root / "preprocessed_shards"
    ckpt_dir = ssd_root / "checkpoints"
    exp_dir = ssd_root / "experiments"
    log_file = exp_dir / "local_pipeline.log"
    state_file = exp_dir / "pipeline_state.json"

    for d in [shards_dir, ckpt_dir, exp_dir, ssd_root / "local_cache"]:
        d.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {}
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))

    def mark_stage(name: str, status: str, extra: Dict[str, Any] | None = None) -> None:
        state[name] = {
            "status": status,
            "updated_at": datetime.datetime.now().isoformat(),
            **(extra or {}),
        }
        atomic_write_json(state_file, state)

    # Phase 0: Crash-resilient Drive -> SSD Sync
    if not args.skip_sync and state.get("drive_sync", {}).get("status") != "completed":
        mark_stage("drive_sync", "in_progress")
        sync_from_drive(ssd_root, args.v2_drive_id, log_file)
        mark_stage("drive_sync", "completed")
    else:
        log_msg("[PHASE 0] Drive sync already completed or skipped.", log_file)

    if args.sync_only:
        log_msg("[SYNC-ONLY] Drive sync finished cleanly. Exiting.", log_file)
        return

    cfg = load_config(REPO_V2 / "configs" / "v2_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    log_msg(
        f"Active Hardware -> Device: {runtime.device} | GPU: {runtime.gpu_name} | "
        f"Tier: {runtime.tier} | Batch: {runtime.batch_size}x{runtime.grad_accum}",
        log_file,
    )

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)
    log_msg(f"Manifest -> Total: {len(manifest_df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(
        drive_shards_root=shards_dir,
        local_cache_dir=shards_dir,
        max_cache_gb=100.0,
    )

    num_workers = 0 if runtime.device in ("cpu", "mps") else int(cfg["storage"]["num_workers"])
    pin_memory = runtime.device == "cuda"

    # Stage 1: SSL Pretraining
    if state.get("stage1_ssl", {}).get("status") != "completed":
        mark_stage("stage1_ssl", "in_progress")
        log_msg("=== Starting / Resuming Stage-1 SSL Pretraining ===", log_file)
        train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.25)
        val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=0.0)
        sampler = BalancedModalityBatchSampler(
            train_df["dataset"].tolist(),
            batch_size=runtime.batch_size,
            temperature_alpha=cfg["training"]["stage2_multitask"]["sampling_temperature"],
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model = PanOrganNetV2.from_config(cfg)
        ssl_cfg = cfg["training"]["stage1_ssl"]
        trainer = ColabResilientTrainer(
            model=model,
            runtime=runtime,
            checkpoint_dir=ckpt_dir / "stage1_ssl_ma_ugm",
            lr=float(ssl_cfg["lr"]),
            weight_decay=float(ssl_cfg["weight_decay"]),
            sacr_loss_weight=float(ssl_cfg["sacr_loss_weight"]),
            moe_aux_weight=float(cfg["model"]["moe_aux_weight"]),
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
        trainer.fit(
            train_loader,
            val_loader,
            num_epochs=int(ssl_cfg["epochs"]),
            mode="ssl",
            warmup_epochs=int(ssl_cfg["warmup_epochs"]),
            resume=True,
        )
        mark_stage("stage1_ssl", "completed", {"best_val_loss": trainer.best_val_loss})
        log_msg(f"Stage-1 SSL Complete! Best Val Loss: {trainer.best_val_loss:.6f}", log_file)

    # Stage 2: Multi-Task Training
    if state.get("stage2_multitask", {}).get("status") != "completed":
        mark_stage("stage2_multitask", "in_progress")
        log_msg("=== Starting / Resuming Stage-2 Multi-Task Training ===", log_file)
        train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
        val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=0.0)
        sampler = BalancedModalityBatchSampler(
            train_df["dataset"].tolist(),
            batch_size=runtime.batch_size,
            temperature_alpha=cfg["training"]["stage2_multitask"]["sampling_temperature"],
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model = PanOrganNetV2.from_config(cfg)
        s2_ckpt_dir = ckpt_dir / "stage2_panorgan_v2_full"
        if not (s2_ckpt_dir / "latest.pt").exists():
            ssl_best = ckpt_dir / "stage1_ssl_ma_ugm" / "best.pt"
            if ssl_best.exists():
                st = torch.load(ssl_best, map_location="cpu", weights_only=False)
                model.load_state_dict(st["model_state_dict"], strict=False)
                log_msg("Initialized Stage-2 from Stage-1 SSL best.pt", log_file)

        s2 = cfg["training"]["stage2_multitask"]
        trainer = ColabResilientTrainer(
            model=model,
            runtime=runtime,
            checkpoint_dir=s2_ckpt_dir,
            lr=float(s2["lr"]),
            backbone_lr_mult=float(s2["backbone_lr_mult"]),
            weight_decay=float(s2["weight_decay"]),
            max_grad_norm=float(s2["max_grad_norm"]),
            seg_loss_weight=float(s2["seg_loss_weight"]),
            cls_loss_weight=float(s2["cls_loss_weight"]),
            sacr_loss_weight=float(s2["sacr_loss_weight"]),
            checkpoint_every_steps=args.checkpoint_every_steps,
            patience=int(s2["patience"]),
        )
        trainer.fit(train_loader, val_loader, num_epochs=int(s2["epochs"]), mode="multitask", resume=True)
        mark_stage("stage2_multitask", "completed", {"best_val_loss": trainer.best_val_loss})
        log_msg(f"Stage-2 Multi-Task Complete! Best Val Loss: {trainer.best_val_loss:.6f}", log_file)

    # Stage 3: Controlled Ablations
    ablations = {
        "ablation_A1_random_mask": {"masking_mode": "random", "use_moe": True, "use_sacr": True},
        "ablation_A2_grad_sgm": {"masking_mode": "gradient_sgm", "use_moe": True, "use_sacr": True},
        "ablation_A3_no_moe": {"masking_mode": "ma_ugm", "use_moe": False, "use_sacr": True},
        "ablation_A4_no_sacr": {"masking_mode": "ma_ugm", "use_moe": True, "use_sacr": False},
    }
    ab_summary: Dict[str, Any] = {}
    for exp_name, overrides in ablations.items():
        if state.get(exp_name, {}).get("status") == "completed":
            ab_summary[exp_name] = state[exp_name]
            continue
        mark_stage(exp_name, "in_progress")
        log_msg(f"=== Starting / Resuming {exp_name} ===", log_file)
        ab_cfg = copy.deepcopy(cfg)
        ab_cfg["model"].update(overrides)
        train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
        val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
        sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=runtime.batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model = PanOrganNetV2.from_config(ab_cfg)
        trainer = ColabResilientTrainer(
            model=model,
            runtime=runtime,
            checkpoint_dir=ckpt_dir / exp_name,
            lr=float(cfg["training"]["stage2_multitask"]["lr"]),
            checkpoint_every_steps=args.checkpoint_every_steps,
            patience=5,
        )
        trainer.fit(train_loader, val_loader, num_epochs=15, mode="multitask", resume=True)
        ab_summary[exp_name] = {"best_val_loss": trainer.best_val_loss}
        mark_stage(exp_name, "completed", {"best_val_loss": trainer.best_val_loss})
    atomic_write_json(exp_dir / "ablation_training_summary.json", ab_summary)

    # Stage 4: Q1 Final Evaluation
    if state.get("eval_q1", {}).get("status") != "completed":
        mark_stage("eval_q1", "in_progress")
        log_msg("=== Running Q1 Evaluation & Bootstrap CIs ===", log_file)
        model = PanOrganNetV2.from_config(cfg).to(runtime.device)
        best_ckpt = ckpt_dir / "stage2_panorgan_v2_full" / "best.pt"
        st = torch.load(best_ckpt, map_location=runtime.device, weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=False)
        model.eval()

        @torch.no_grad()
        def eval_split(df_sub: pd.DataFrame, mnar_trunc: float = 0.0):
            ds = PanOrganShardDataset(df_sub, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=mnar_trunc)
            loader = DataLoader(ds, batch_size=runtime.batch_size, shuffle=False, num_workers=num_workers)
            seg_records: Dict[str, list] = {}
            for batch in loader:
                out = model(
                    batch["image"].to(runtime.device),
                    batch["meta_vec"].to(runtime.device),
                    batch["is_3d"].to(runtime.device),
                    prompts=batch["prompt"],
                )
                has_mask = batch["has_mask"].numpy()
                preds_mask = out["seg_logits"].argmax(dim=1).cpu().numpy()
                true_mask = batch["mask"].numpy()
                for i in range(len(has_mask)):
                    if has_mask[i]:
                        res = compute_dice_and_hd95(preds_mask[i] > 0, true_mask[i] > 0, spacing_mm=(1.5, 1.5))
                        seg_records.setdefault(batch["dataset"][i], []).append(res)
            seg_summary = {}
            for ds_name, items in seg_records.items():
                dices = np.array([x["dice"] for x in items])
                hd95s = np.array([x["hd95_mm"] for x in items])
                seg_summary[ds_name] = {
                    "dice_ci": bootstrap_confidence_interval(dices, n_bootstrap=cfg["evaluation"]["bootstrap_iterations"]),
                    "hd95_ci": bootstrap_confidence_interval(hd95s, n_bootstrap=cfg["evaluation"]["bootstrap_iterations"]),
                    "n_samples": len(items),
                }
            return seg_summary

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_segmentation": eval_split(test_df, 0.0),
            "mnar_30_segmentation": eval_split(test_df, 0.30),
            "checkpoint_evaluated": str(best_ckpt),
        }
        atomic_write_json(exp_dir / "q1_final_evaluation_report.json", report)
        mark_stage("eval_q1", "completed")
        log_msg("=== ALL STAGES COMPLETED SUCCESSFULLY ===", log_file)


if __name__ == "__main__":
    main()
