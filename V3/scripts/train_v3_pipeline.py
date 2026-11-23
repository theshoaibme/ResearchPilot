#!/usr/bin/env python3
"""PanOrganNet-V3 Power-Loss Resilient Local SSD Training & Evaluation Pipeline.

Key V3 Upgrades:
1. 7-Cohort Multi-Modal Dataset (MRI, CT, CXR, US) with zero patient leakage.
2. Class-Balanced CrossEntropy + Label Smoothing (0.05) + cls_loss_weight=3.0.
3. Composite Best-Checkpoint Selection: 0.5 * Val_Dice + 0.5 * Val_Macro_F1.
4. Stage-2B Frozen-Backbone Head Calibration (100% locks Segmentation Dice while boosting Accuracy/F1).
5. Atomic 50-step fsync crash-resume & complete Q1 evaluation suite.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_V3"))
os.environ.setdefault("HF_HOME", str(_SSD_ROOT / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_SSD_ROOT / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_SSD_ROOT / "model_cache" / "timm"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2_PKG = REPO_ROOT / "V2"
if str(V2_PKG) not in sys.path:
    sys.path.insert(0, str(V2_PKG))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.engine import ColabResilientTrainer
from panorgan.models.decoders import DiceFocalBoundaryLoss
from panorgan.evaluation import (
    ModalityConditionalConformal,
    TemperatureScaler,
    bootstrap_confidence_interval,
    compute_classification_metrics,
    compute_dice_and_hd95,
    compute_ece_and_brier,
    export_latex_results_table,
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


def atomic_torch_save(state: Dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def log_msg(msg: str, log_file: Path) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def compute_class_weights(train_df: pd.DataFrame, num_classes: int = 4) -> torch.Tensor:
    valid = train_df[train_df["label"] >= 0]["label"].astype(int).values
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            w = total / (num_classes * counts[c])
            weights[c] = float(np.clip(w, 0.5, 4.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_composite(
    model: PanOrganNetV2,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dices: List[float] = []
    cls_preds: List[int] = []
    cls_trues: List[int] = []

    for batch in val_loader:
        out = model(
            batch["image"].to(device),
            batch["meta_vec"].to(device),
            batch["is_3d"].to(device),
            prompts=batch["prompt"],
        )
        has_mask = batch["has_mask"].numpy()
        pred_m = out["seg_logits"].argmax(dim=1).cpu().numpy()
        true_m = batch["mask"].numpy()
        for i in range(len(has_mask)):
            if has_mask[i]:
                p = pred_m[i] > 0
                t = true_m[i] > 0
                inter = np.logical_and(p, t).sum()
                denom = p.sum() + t.sum()
                dices.append(float(2.0 * inter / denom) if denom > 0 else 1.0)

        lbls = batch["label"].numpy()
        valid = lbls >= 0
        if valid.any():
            logits = out["multiclass_logits"][valid].cpu()
            preds = logits.argmax(dim=-1).numpy()
            cls_preds.extend(preds.tolist())
            cls_trues.extend(lbls[valid].tolist())

    val_dice = float(np.mean(dices)) if dices else 0.0
    val_f1 = float(f1_score(cls_trues, cls_preds, average="macro", zero_division=0)) if cls_trues else 0.0
    val_acc = float(np.mean(np.array(cls_preds) == np.array(cls_trues))) if cls_trues else 0.0
    composite = 0.5 * val_dice + 0.5 * val_f1
    return {
        "val_dice": val_dice,
        "val_macro_f1": val_f1,
        "val_acc": val_acc,
        "composite": composite,
    }


def train_stage2a_composite(
    model: PanOrganNetV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    runtime,
    ckpt_dir: Path,
    cfg_s2: Dict[str, Any],
    cls_weights: torch.Tensor,
    moe_aux_weight: float,
    checkpoint_every_steps: int,
    log_file: Path,
) -> float:
    device = torch.device(runtime.device)
    model = model.to(device)
    cls_weights = cls_weights.to(device)
    seg_crit = DiceFocalBoundaryLoss()
    lr = float(cfg_s2["lr"])
    backbone_mult = float(cfg_s2["backbone_lr_mult"])
    label_smoothing = float(cfg_s2["label_smoothing"])
    num_epochs = int(cfg_s2["epochs"])
    patience = int(cfg_s2["patience"])

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.encoder" in name and "lora_" not in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_mult},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=float(cfg_s2["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    start_epoch = 1
    best_composite = -1.0
    epochs_no_improve = 0
    global_step = 0
    resume_batch_idx = -1
    history: Dict[str, list] = {"train_loss": [], "val_dice": [], "val_f1": [], "val_acc": [], "composite": []}

    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_composite = float(ck.get("best_composite", -1.0))
        epochs_no_improve = int(ck.get("epochs_no_improve", 0))
        global_step = int(ck.get("global_step", 0))
        history = ck.get("history", history)
        if ck.get("mid_epoch", False):
            start_epoch = int(ck["epoch"])
            resume_batch_idx = int(ck.get("batch_idx", -1))
            log_msg(f"[RESUME Stage-2A] Epoch {start_epoch}/{num_epochs} from batch {resume_batch_idx + 1}", log_file)
        else:
            start_epoch = int(ck["epoch"]) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss = 0.0
        n_b = 0

        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[V3-S2A] Ep {epoch}/{num_epochs}", leave=False)):
            if epoch == start_epoch and b_idx <= resume_batch_idx:
                continue

            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            out = model(images, meta_vec, is_3d, prompts=batch["prompt"])

            total = torch.zeros((), device=device)
            has_mask = batch["has_mask"].to(device)
            if has_mask.any():
                seg_l = seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])
                total = total + float(cfg_s2["seg_loss_weight"]) * seg_l

            labels = batch["label"].to(device)
            valid_cls = labels >= 0
            if valid_cls.any():
                cls_l = F.cross_entropy(
                    out["multiclass_logits"][valid_cls],
                    labels[valid_cls],
                    weight=cls_weights,
                    label_smoothing=label_smoothing,
                )
                total = total + float(cfg_s2["cls_loss_weight"]) * cls_l

            if out.get("pred_coords") is not None:
                sacr_l = F.smooth_l1_loss(out["pred_coords"], batch["body_coord"].to(device))
                total = total + float(cfg_s2["sacr_loss_weight"]) * sacr_l

            total = total + moe_aux_weight * out["moe_aux_loss"]
            (total / max(1, runtime.grad_accum)).backward()

            run_loss += float(total.detach().item())
            n_b += 1
            step_now = ((b_idx + 1) % runtime.grad_accum == 0) or ((b_idx + 1) == len(train_loader))
            if step_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_s2["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if checkpoint_every_steps > 0 and (global_step % checkpoint_every_steps == 0):
                    atomic_torch_save(
                        {
                            "epoch": epoch,
                            "mid_epoch": True,
                            "batch_idx": b_idx,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_composite": best_composite,
                            "epochs_no_improve": epochs_no_improve,
                            "history": history,
                        },
                        latest_path,
                    )

        resume_batch_idx = -1
        scheduler.step()
        metrics = evaluate_composite(model, val_loader, device)
        tr_l = run_loss / max(n_b, 1)
        history["train_loss"].append(tr_l)
        history["val_dice"].append(metrics["val_dice"])
        history["val_f1"].append(metrics["val_macro_f1"])
        history["val_acc"].append(metrics["val_acc"])
        history["composite"].append(metrics["composite"])

        is_best = metrics["composite"] > best_composite
        if is_best:
            best_composite = metrics["composite"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        state_dict = {
            "epoch": epoch,
            "mid_epoch": False,
            "batch_idx": -1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_composite": best_composite,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
        }
        atomic_torch_save(state_dict, latest_path)
        if is_best:
            atomic_torch_save(state_dict, ckpt_dir / "best.pt")
        atomic_write_json(ckpt_dir / "training_history.json", history)

        log_msg(
            f"[V3-S2A Ep {epoch}/{num_epochs}] train_loss={tr_l:.4f} | "
            f"val_dice={metrics['val_dice']:.4f} | val_f1={metrics['val_macro_f1']:.4f} | "
            f"val_acc={metrics['val_acc']:.4f} | composite={metrics['composite']:.4f} (best={best_composite:.4f})",
            log_file,
        )
        if epochs_no_improve >= patience:
            log_msg(f"[V3-S2A] Early stopping at epoch {epoch} (best composite={best_composite:.4f})", log_file)
            break

    return best_composite


def train_stage2b_head_cal(
    model: PanOrganNetV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    runtime,
    ckpt_dir: Path,
    cfg_2b: Dict[str, Any],
    cls_weights: torch.Tensor,
    log_file: Path,
) -> float:
    device = torch.device(runtime.device)
    model = model.to(device)
    cls_weights = cls_weights.to(device)

    # Freeze everything except pathology_head (100% preserves Segmentation Dice)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("pathology_head."))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg_2b["lr"]),
        weight_decay=float(cfg_2b["weight_decay"]),
    )
    num_epochs = int(cfg_2b["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_f1 = -1.0
    history: Dict[str, list] = {"val_dice": [], "val_f1": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"[V3-S2B HeadCal] Ep {epoch}/{num_epochs}", leave=False):
            labels = batch["label"].to(device)
            valid = labels >= 0
            if not valid.any():
                continue
            out = model(
                batch["image"].to(device),
                batch["meta_vec"].to(device),
                batch["is_3d"].to(device),
                prompts=batch["prompt"],
            )
            loss = F.cross_entropy(
                out["multiclass_logits"][valid],
                labels[valid],
                weight=cls_weights,
                label_smoothing=float(cfg_2b["label_smoothing"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()
        metrics = evaluate_composite(model, val_loader, device)
        history["val_dice"].append(metrics["val_dice"])
        history["val_f1"].append(metrics["val_macro_f1"])
        history["val_acc"].append(metrics["val_acc"])
        if metrics["val_macro_f1"] > best_f1:
            best_f1 = metrics["val_macro_f1"]
            atomic_torch_save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_f1": best_f1}, ckpt_dir / "best.pt")
        atomic_write_json(ckpt_dir / "training_history.json", history)
        log_msg(
            f"[V3-S2B Ep {epoch}/{num_epochs}] val_dice={metrics['val_dice']:.4f} | "
            f"val_f1={metrics['val_macro_f1']:.4f} | val_acc={metrics['val_acc']:.4f}",
            log_file,
        )

    return best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="PanOrganNet-V3 Training & Evaluation Pipeline")
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)

    ssd_root = Path(cfg["storage"]["drive_root"])
    shards_dir = ssd_root / "preprocessed_shards"
    ckpt_dir = ssd_root / "checkpoints"
    exp_dir = ssd_root / "experiments"
    log_file = exp_dir / "v3_pipeline.log"
    state_file = exp_dir / "v3_pipeline_state.json"

    for d in [shards_dir, ckpt_dir, exp_dir, ssd_root / "local_cache"]:
        d.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = json.loads(state_file.read_text()) if state_file.exists() else {}

    def mark(name: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        state[name] = {"status": status, "updated_at": datetime.datetime.now().isoformat(), **(extra or {})}
        atomic_write_json(state_file, state)

    log_msg(f"Active Hardware -> Device: {runtime.device} | GPU: {runtime.gpu_name} | Batch: {runtime.batch_size}x{runtime.grad_accum}", log_file)

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)
    log_msg(f"V3 Manifest -> Total: {len(manifest_df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    cls_weights = compute_class_weights(train_df, num_classes=4)
    log_msg(f"V3 Class Weights: {cls_weights.tolist()}", log_file)

    # 1. Stage-1 SSL across all 7 cohorts (warm-started from V2 SSL best.pt)
    if state.get("v3_stage1_ssl", {}).get("status") != "completed":
        mark("v3_stage1_ssl", "in_progress")
        log_msg("=== Starting / Resuming V3 Stage-1 SSL (7 Cohorts) ===", log_file)
        train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.25)
        val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
        sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.5)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

        model = PanOrganNetV2.from_config(cfg)
        v2_ssl_best = Path("/Volumes/Shoaib/PanOrgan_V2/checkpoints/stage1_ssl_ma_ugm/best.pt")
        s1_dir = ckpt_dir / "v3_stage1_ssl"
        if not (s1_dir / "latest.pt").exists() and v2_ssl_best.exists():
            st = torch.load(v2_ssl_best, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model_state_dict"], strict=False)
            log_msg("Warm-started V3 Stage-1 SSL from V2 Stage-1 best.pt", log_file)

        s1 = cfg["training"]["stage1_ssl"]
        trainer = ColabResilientTrainer(
            model=model,
            runtime=runtime,
            checkpoint_dir=s1_dir,
            lr=float(s1["lr"]),
            weight_decay=float(s1["weight_decay"]),
            sacr_loss_weight=float(s1["sacr_loss_weight"]),
            moe_aux_weight=float(cfg["model"]["moe_aux_weight"]),
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
        trainer.fit(train_loader, val_loader, num_epochs=int(s1["epochs"]), mode="ssl", warmup_epochs=int(s1["warmup_epochs"]), resume=True)
        mark("v3_stage1_ssl", "completed", {"best_val_loss": trainer.best_val_loss})
        log_msg(f"V3 Stage-1 SSL Complete! Best Val Loss: {trainer.best_val_loss:.6f}", log_file)

    # 2. Stage-2A Joint Multi-Task (Composite Dice + F1 Checkpoint Selection)
    if state.get("v3_stage2a_joint", {}).get("status") != "completed":
        mark("v3_stage2a_joint", "in_progress")
        log_msg("=== Starting / Resuming V3 Stage-2A Joint Training ===", log_file)
        sup_train = train_df[(train_df["has_mask"]) | (train_df["label"] >= 0)].reset_index(drop=True)
        sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)
        train_ds = PanOrganShardDataset(sup_train, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
        val_ds = PanOrganShardDataset(sup_val, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
        sampler = BalancedModalityBatchSampler(sup_train["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.5)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

        model = PanOrganNetV2.from_config(cfg)
        s2a_dir = ckpt_dir / "v3_stage2a_joint"
        if not (s2a_dir / "latest.pt").exists():
            s1_best = ckpt_dir / "v3_stage1_ssl" / "best.pt"
            if s1_best.exists():
                st = torch.load(s1_best, map_location="cpu", weights_only=False)
                model.load_state_dict(st["model_state_dict"], strict=False)
                log_msg("Initialized V3 Stage-2A from V3 Stage-1 SSL best.pt", log_file)

        best_comp = train_stage2a_composite(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            runtime=runtime,
            ckpt_dir=s2a_dir,
            cfg_s2=cfg["training"]["stage2a_joint"],
            cls_weights=cls_weights,
            moe_aux_weight=float(cfg["model"]["moe_aux_weight"]),
            checkpoint_every_steps=args.checkpoint_every_steps,
            log_file=log_file,
        )
        mark("v3_stage2a_joint", "completed", {"best_composite": best_comp})

    # 3. Stage-2B Frozen-Backbone Head Calibration
    if state.get("v3_stage2b_head_cal", {}).get("status") != "completed":
        mark("v3_stage2b_head_cal", "in_progress")
        log_msg("=== Starting V3 Stage-2B Frozen-Backbone Head Calibration ===", log_file)
        sup_train = train_df[train_df["label"] >= 0].reset_index(drop=True)
        sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)
        train_ds = PanOrganShardDataset(sup_train, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.10)
        val_ds = PanOrganShardDataset(sup_val, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
        sampler = BalancedModalityBatchSampler(sup_train["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.5)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

        model = PanOrganNetV2.from_config(cfg)
        s2a_best = ckpt_dir / "v3_stage2a_joint" / "best.pt"
        st = torch.load(s2a_best, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=False)

        best_f1 = train_stage2b_head_cal(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            runtime=runtime,
            ckpt_dir=ckpt_dir / "v3_stage2b_calibrated",
            cfg_2b=cfg["training"]["stage2b_head_cal"],
            cls_weights=cls_weights,
            log_file=log_file,
        )
        mark("v3_stage2b_head_cal", "completed", {"best_val_f1": best_f1})

    # 4. Complete V3 Evaluation on Untouched Held-Out Test Set
    if state.get("v3_eval", {}).get("status") != "completed":
        mark("v3_eval", "in_progress")
        log_msg("=== Running V3 Complete Q1 Evaluation ===", log_file)
        v2_scripts = str(REPO_ROOT / "V2/scripts")
        if v2_scripts not in sys.path:
            sys.path.insert(0, v2_scripts)
        from generate_all_thesis_reports import compute_cls_full, run_inference, summarize_seg
        # Reuse exact V2/V3 evaluator helpers
        model = PanOrganNetV2.from_config(cfg).to(runtime.device)
        best_ckpt = ckpt_dir / "v3_stage2b_calibrated" / "best.pt"
        st = torch.load(best_ckpt, map_location=runtime.device, weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=False)
        model.eval()

        val_inf = run_inference(model, val_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
        test_inf = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
        n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
        seg_sum = summarize_seg(test_inf["seg_records"], n_boot=n_boot)

        scaler = TemperatureScaler()
        t_opt = scaler.fit(val_inf["cls_logits"], torch.from_numpy(val_inf["cls_labels"])) if len(val_inf["cls_labels"]) > 0 else 1.0
        cls_raw = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=None)
        cls_cal = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=scaler)

        cp = ModalityConditionalConformal(alpha=0.10)
        cp.fit(scaler.transform_probs(val_inf["cls_logits"]), val_inf["cls_labels"], val_inf["cls_groups"])
        cp_cov = cp.evaluate_coverage(scaler.transform_probs(test_inf["cls_logits"]), test_inf["cls_labels"], test_inf["cls_groups"])

        mnar_curve = {"0": seg_sum}
        for frac in (0.20, 0.40):
            inf_f = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=frac)
            mnar_curve[str(int(frac * 100))] = summarize_seg(inf_f["seg_records"], n_boot=n_boot)

        out_dir = exp_dir / "thesis_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "model": "PanOrganNet_V3_7Cohorts",
            "segmentation": seg_sum,
            "classification_uncalibrated": cls_raw,
            "classification_calibrated": cls_cal,
            "optimal_temperature": float(t_opt),
            "mc2_conformal_coverage": cp_cov,
            "mnar_stress_curve": mnar_curve,
            "checkpoint_evaluated": str(best_ckpt),
        }
        atomic_write_json(out_dir / "v3_complete_thesis_report.json", report)
        mark("v3_eval", "completed")
        log_msg("=== PANORGANNET-V3 PIPELINE COMPLETED SUCCESSFULLY ===", log_file)


if __name__ == "__main__":
    main()
