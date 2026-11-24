#!/usr/bin/env python3
"""PanOrgan 3.2 Balanced 5-Modality Upgrade with Early Stopping, MixUp, and Best-Checkpoint Selection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "V2"))
sys.path.insert(0, str(REPO_ROOT / "V2/scripts"))
sys.path.insert(0, str(REPO_ROOT / "V3/scripts"))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.evaluation import ModalityConditionalConformal, TemperatureScaler
from panorgan.models import PanOrganNetV2
from panorgan.models.decoders import DiceFocalBoundaryLoss
from generate_all_thesis_reports import compute_cls_full, run_inference, summarize_seg
from train_v3_pipeline import atomic_torch_save, atomic_write_json, compute_class_weights, log_msg
from train_v31_upgrade import evaluate_with_loss


def main() -> None:
    os.environ.setdefault("HF_HOME", "/Volumes/Shoaib/PanOrgan_V3/model_cache/hf")
    os.environ.setdefault("TORCH_HOME", "/Volumes/Shoaib/PanOrgan_V3/model_cache/torch")
    os.environ.setdefault("TIMM_CACHE_DIR", "/Volumes/Shoaib/PanOrgan_V3/model_cache/timm")

    parser = argparse.ArgumentParser(description="PanOrgan 3.2 Balanced Upgrade")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    ssd_root = Path(cfg["storage"]["drive_root"])
    shards_dir = ssd_root / "preprocessed_shards"
    ckpt_dir = ssd_root / "checkpoints" / "v32_balanced"
    exp_dir = ssd_root / "experiments"
    log_file = exp_dir / "v32_pipeline.log"
    state_file = exp_dir / "v32_pipeline_state.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = json.loads(state_file.read_text()) if state_file.exists() else {}

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    sup_train = train_df[(train_df["has_mask"]) | (train_df["label"] >= 0)].reset_index(drop=True)
    sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)
    log_msg(f"PanOrgan 3.2 Supervised -> Train: {len(sup_train)} | Val: {len(sup_val)} | Test: {len(test_df)}", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    cls_weights = compute_class_weights(sup_train, num_classes=4).to(device)
    seg_crit = DiceFocalBoundaryLoss()

    train_ds = PanOrganShardDataset(sup_train, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
    val_ds = PanOrganShardDataset(sup_val, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    sampler = BalancedModalityBatchSampler(sup_train["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.35)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

    model = PanOrganNetV2.from_config(cfg).to(device)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    if state.get("v32_train", {}).get("status") != "completed":
        start_epoch = 1
        best_comp = -1.0
        no_improve = 0
        resume_b = -1
        global_step = 0
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dice": [], "val_f1": [], "val_acc": [], "composite": []}

        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        if latest_path.exists():
            ck = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            best_comp = float(ck.get("best_composite", -1.0))
            no_improve = int(ck.get("no_improve", 0))
            global_step = int(ck.get("global_step", 0))
            history = ck.get("history", history)
            if ck.get("mid_epoch", False):
                start_epoch = int(ck["epoch"])
                resume_b = int(ck.get("batch_idx", -1))
                log_msg(f"[RESUME 3.2] Epoch {start_epoch}/{args.epochs} from batch {resume_b + 1}", log_file)
            else:
                start_epoch = int(ck["epoch"]) + 1
        else:
            v31_best = ssd_root / "checkpoints" / "v31_5modal_mixup" / "best.pt"
            st = torch.load(v31_best, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model_state_dict"], strict=False)
            log_msg(f"Warm-started PanOrgan 3.2 from {v31_best}", log_file)

        rng = np.random.default_rng(42)
        for epoch in range(start_epoch, args.epochs + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            run_loss = 0.0
            n_b = 0

            for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[PanOrgan 3.2] Ep {epoch}/{args.epochs}", leave=False)):
                if epoch == start_epoch and b_idx <= resume_b:
                    continue

                images = batch["image"].to(device)
                meta_vec = batch["meta_vec"].to(device)
                is_3d = batch["is_3d"].to(device)
                has_mask = batch["has_mask"].to(device)
                labels = batch["label"].to(device)
                valid_cls = labels >= 0

                use_mixup = (not has_mask.any()) and valid_cls.all() and (images.size(0) >= 2) and (rng.random() < 0.25)
                if use_mixup:
                    lam = float(rng.beta(0.2, 0.2))
                    perm = torch.randperm(images.size(0), device=device)
                    images_in = lam * images + (1.0 - lam) * images[perm]
                else:
                    lam, perm, images_in = 1.0, None, images

                out = model(images_in, meta_vec, is_3d, prompts=batch["prompt"])
                total = torch.zeros((), device=device)

                if has_mask.any():
                    total = total + 1.0 * seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])

                if valid_cls.any():
                    if use_mixup and perm is not None:
                        c1 = F.cross_entropy(out["multiclass_logits"], labels, weight=cls_weights, label_smoothing=0.05)
                        c2 = F.cross_entropy(out["multiclass_logits"], labels[perm], weight=cls_weights, label_smoothing=0.05)
                        cls_l = lam * c1 + (1.0 - lam) * c2
                    else:
                        cls_l = F.cross_entropy(out["multiclass_logits"][valid_cls], labels[valid_cls], weight=cls_weights, label_smoothing=0.05)
                    total = total + 3.0 * cls_l

                if out.get("pred_coords") is not None:
                    total = total + 0.25 * F.smooth_l1_loss(out["pred_coords"], batch["body_coord"].to(device))
                total = total + float(cfg["model"]["moe_aux_weight"]) * out["moe_aux_loss"]

                (total / max(1, runtime.grad_accum)).backward()
                run_loss += float(total.detach().item())
                n_b += 1

                if ((b_idx + 1) % runtime.grad_accum == 0) or ((b_idx + 1) == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % args.checkpoint_every_steps == 0:
                        atomic_torch_save({
                            "epoch": epoch, "mid_epoch": True, "batch_idx": b_idx, "global_step": global_step,
                            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(), "best_composite": best_comp,
                            "no_improve": no_improve, "history": history,
                        }, latest_path)

            resume_b = -1
            scheduler.step()
            metrics = evaluate_with_loss(model, val_loader, device, cls_weights, seg_crit)
            tr_l = run_loss / max(n_b, 1)
            history["train_loss"].append(tr_l)
            history["val_loss"].append(metrics["val_loss"])
            history["val_dice"].append(metrics["val_dice"])
            history["val_f1"].append(metrics["val_macro_f1"])
            history["val_acc"].append(metrics["val_acc"])
            history["composite"].append(metrics["composite"])

            is_best = metrics["composite"] > best_comp
            if is_best:
                best_comp = metrics["composite"]
                no_improve = 0
            else:
                no_improve += 1

            st_out = {
                "epoch": epoch, "mid_epoch": False, "batch_idx": -1, "global_step": global_step,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_composite": best_comp,
                "no_improve": no_improve, "history": history,
            }
            atomic_torch_save(st_out, latest_path)
            if is_best:
                atomic_torch_save(st_out, best_path)
            atomic_write_json(ckpt_dir / "training_history.json", history)

            log_msg(
                f"[3.2 Ep {epoch}/{args.epochs}] train_loss={tr_l:.4f} | val_loss={metrics['val_loss']:.4f} | "
                f"val_dice={metrics['val_dice']:.4f} | val_f1={metrics['val_macro_f1']:.4f} | "
                f"val_acc={metrics['val_acc']:.4f} (best={best_comp:.4f}, no_improve={no_improve}/{args.patience})",
                log_file,
            )
            if no_improve >= args.patience:
                log_msg(f"[EARLY STOPPING 3.2] Triggered at Epoch {epoch}; keeping best checkpoint (composite={best_comp:.4f}).", log_file)
                break

        state["v32_train"] = {"status": "completed", "best_composite": best_comp}
        atomic_write_json(state_file, state)

    log_msg("=== Running PanOrgan 3.2 Complete Evaluation ===", log_file)
    st = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()

    val_inf = run_inference(model, val_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
    test_inf = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
    seg_sum = summarize_seg(test_inf["seg_records"], n_boot=n_boot)

    scaler = TemperatureScaler()
    t_opt = scaler.fit(val_inf["cls_logits"], torch.from_numpy(val_inf["cls_labels"]))
    cls_raw = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=None)
    cls_cal = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=scaler)

    cp = ModalityConditionalConformal(alpha=0.10)
    cp.fit(scaler.transform_probs(val_inf["cls_logits"]), val_inf["cls_labels"], val_inf["cls_groups"])
    cp_cov = cp.evaluate_coverage(scaler.transform_probs(test_inf["cls_logits"]), test_inf["cls_labels"], test_inf["cls_groups"])

    out_dir = exp_dir / "thesis_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "PanOrgan_3.2_Balanced_5Modal",
        "segmentation": seg_sum,
        "classification_uncalibrated": cls_raw,
        "classification_calibrated": cls_cal,
        "optimal_temperature": float(t_opt),
        "mc2_conformal_coverage": cp_cov,
        "checkpoint_evaluated": str(best_path),
    }
    atomic_write_json(out_dir / "v32_complete_thesis_report.json", report)
    state["v32_eval"] = {"status": "completed"}
    atomic_write_json(state_file, state)
    log_msg("=== PANORGAN 3.2 COMPLETED SUCCESSFULLY ===", log_file)


if __name__ == "__main__":
    main()
