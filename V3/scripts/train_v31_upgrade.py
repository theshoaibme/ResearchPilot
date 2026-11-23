#!/usr/bin/env python3
"""PanOrganNet-V3.1 Upgrade: Official-Label CBIS-DDSM (MG) + TCGA-KIRC (CT) + MixUp + val_loss Tracking.

Warm-starts from V3 best.pt (zero wasted training), adds MixUp regularization and explicit val_loss
monitoring, fine-tunes across all 5 imaging modalities (MRI, CT, CXR, US, MG), and runs full Q1 eval.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_V3"))
os.environ.setdefault("HF_HOME", str(_SSD_ROOT / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_SSD_ROOT / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_SSD_ROOT / "model_cache" / "timm"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
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


@torch.no_grad()
def evaluate_with_loss(
    model: PanOrganNetV2,
    val_loader: DataLoader,
    device: torch.device,
    cls_weights: torch.Tensor,
    seg_crit: DiceFocalBoundaryLoss,
) -> Dict[str, float]:
    model.eval()
    dices: List[float] = []
    cls_preds: List[int] = []
    cls_trues: List[int] = []
    val_losses: List[float] = []

    for batch in val_loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        out = model(images, meta_vec, is_3d, prompts=batch["prompt"])

        loss = torch.zeros((), device=device)
        has_mask = batch["has_mask"].to(device)
        if has_mask.any():
            loss = loss + 1.0 * seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])

        labels = batch["label"].to(device)
        valid = labels >= 0
        if valid.any():
            loss = loss + 3.0 * F.cross_entropy(out["multiclass_logits"][valid], labels[valid], weight=cls_weights, label_smoothing=0.05)
        val_losses.append(float(loss.item()))

        hm_np = batch["has_mask"].numpy()
        pred_m = out["seg_logits"].argmax(dim=1).cpu().numpy()
        true_m = batch["mask"].numpy()
        for i in range(len(hm_np)):
            if hm_np[i]:
                p, t = pred_m[i] > 0, true_m[i] > 0
                inter = np.logical_and(p, t).sum()
                denom = p.sum() + t.sum()
                dices.append(float(2.0 * inter / denom) if denom > 0 else 1.0)

        lbl_np = batch["label"].numpy()
        v_np = lbl_np >= 0
        if v_np.any():
            preds = out["multiclass_logits"][v_np].cpu().argmax(dim=-1).numpy()
            cls_preds.extend(preds.tolist())
            cls_trues.extend(lbl_np[v_np].tolist())

    val_dice = float(np.mean(dices)) if dices else 0.0
    val_f1 = float(f1_score(cls_trues, cls_preds, average="macro", zero_division=0)) if cls_trues else 0.0
    val_acc = float(np.mean(np.array(cls_preds) == np.array(cls_trues))) if cls_trues else 0.0
    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "val_dice": val_dice,
        "val_macro_f1": val_f1,
        "val_acc": val_acc,
        "composite": 0.5 * val_dice + 0.5 * val_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PanOrganNet-V3.1 5-Modality Upgrade")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    ssd_root = Path(cfg["storage"]["drive_root"])
    shards_dir = ssd_root / "preprocessed_shards"
    ckpt_dir = ssd_root / "checkpoints" / "v31_5modal_mixup"
    exp_dir = ssd_root / "experiments"
    log_file = exp_dir / "v31_pipeline.log"
    state_file = exp_dir / "v31_pipeline_state.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = json.loads(state_file.read_text()) if state_file.exists() else {}

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    sup_train = train_df[(train_df["has_mask"]) | (train_df["label"] >= 0)].reset_index(drop=True)
    sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)
    log_msg(f"V3.1 Supervised -> Train: {len(sup_train)} | Val: {len(sup_val)} | Test: {len(test_df)}", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    cls_weights = compute_class_weights(sup_train, num_classes=4).to(device)
    seg_crit = DiceFocalBoundaryLoss()

    train_ds = PanOrganShardDataset(sup_train, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
    val_ds = PanOrganShardDataset(sup_val, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    sampler = BalancedModalityBatchSampler(sup_train["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.5)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

    model = PanOrganNetV2.from_config(cfg).to(device)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    if state.get("v31_train", {}).get("status") != "completed":
        start_epoch = 1
        best_comp = -1.0
        resume_b = -1
        global_step = 0
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dice": [], "val_f1": [], "val_acc": [], "composite": []}

        optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-5, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        if latest_path.exists():
            ck = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            best_comp = float(ck.get("best_composite", -1.0))
            global_step = int(ck.get("global_step", 0))
            history = ck.get("history", history)
            if ck.get("mid_epoch", False):
                start_epoch = int(ck["epoch"])
                resume_b = int(ck.get("batch_idx", -1))
                log_msg(f"[RESUME V3.1] Epoch {start_epoch}/{args.epochs} from batch {resume_b + 1}", log_file)
            else:
                start_epoch = int(ck["epoch"]) + 1
        else:
            v3_best = ssd_root / "checkpoints" / "v3_stage2b_calibrated" / "best.pt"
            st = torch.load(v3_best, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model_state_dict"], strict=False)
            log_msg(f"Warm-started V3.1 from {v3_best}", log_file)

        rng = np.random.default_rng(42)
        for epoch in range(start_epoch, args.epochs + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            run_loss = 0.0
            n_b = 0

            for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[V3.1 5-Modal+MixUp] Ep {epoch}/{args.epochs}", leave=False)):
                if epoch == start_epoch and b_idx <= resume_b:
                    continue

                images = batch["image"].to(device)
                meta_vec = batch["meta_vec"].to(device)
                is_3d = batch["is_3d"].to(device)
                has_mask = batch["has_mask"].to(device)
                labels = batch["label"].to(device)
                valid_cls = labels >= 0

                # Safe MixUp (alpha=0.2, p=0.25) on classification-only samples
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
                            "scheduler_state_dict": scheduler.state_dict(), "best_composite": best_comp, "history": history,
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
            st_out = {
                "epoch": epoch, "mid_epoch": False, "batch_idx": -1, "global_step": global_step,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_composite": best_comp, "history": history,
            }
            atomic_torch_save(st_out, latest_path)
            if is_best:
                atomic_torch_save(st_out, best_path)
            atomic_write_json(ckpt_dir / "training_history.json", history)

            log_msg(
                f"[V3.1 Ep {epoch}/{args.epochs}] train_loss={tr_l:.4f} | val_loss={metrics['val_loss']:.4f} | "
                f"val_dice={metrics['val_dice']:.4f} | val_f1={metrics['val_macro_f1']:.4f} | val_acc={metrics['val_acc']:.4f} (best={best_comp:.4f})",
                log_file,
            )

        state["v31_train"] = {"status": "completed", "best_composite": best_comp}
        atomic_write_json(state_file, state)

    # Complete Q1 Evaluation across all 5 modalities
    log_msg("=== Running V3.1 Complete 5-Modality Q1 Evaluation ===", log_file)
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
        "model": "PanOrganNet_V3.1_5Modal_7SupCohorts",
        "segmentation": seg_sum,
        "classification_uncalibrated": cls_raw,
        "classification_calibrated": cls_cal,
        "optimal_temperature": float(t_opt),
        "mc2_conformal_coverage": cp_cov,
        "checkpoint_evaluated": str(best_path),
    }
    atomic_write_json(out_dir / "v31_complete_thesis_report.json", report)
    state["v31_eval"] = {"status": "completed"}
    atomic_write_json(state_file, state)
    log_msg("=== PANORGANNET-V3.1 COMPLETED SUCCESSFULLY ===", log_file)


if __name__ == "__main__":
    main()
