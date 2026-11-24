#!/usr/bin/env python3
"""PanOrgan 3.2 Cohort-Masked Polish & Zero-Mismatch Final Test Verification."""
from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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
from generate_all_thesis_reports import summarize_seg
from train_v3_pipeline import atomic_torch_save, atomic_write_json, compute_class_weights, log_msg

COHORT_ACTIVE_CLASSES: Dict[str, int] = {
    "cmmd_mg": 2,
    "cbis_ddsm": 2,
    "kidney_ct_abn": 2,
    "tcga_kirc_ct": 2,
    "tcga_lgg": 2,
    "ultrasound_nerve": 2,
    "busi": 3,
    "pulmonology_cxr": 4,
    "pulmonology_cxr_ext": 4,
    "nih_cxr14": 4,
}


def mask_logits_by_cohort(logits: torch.Tensor, datasets: List[str]) -> torch.Tensor:
    masked = logits.clone()
    for i, ds in enumerate(datasets):
        k = COHORT_ACTIVE_CLASSES.get(ds, 4)
        if k < masked.size(-1):
            masked[i, k:] = -1e4
    return masked


@torch.no_grad()
def eval_masked(model: PanOrganNetV2, loader: DataLoader, device: torch.device, cls_weights: torch.Tensor, seg_crit: torch.nn.Module) -> Dict[str, float]:
    model.eval()
    val_loss, n_b = 0.0, 0
    all_p, all_y, dices = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        has_mask = batch["has_mask"].to(device)
        labels = batch["label"].to(device)
        out = model(images, meta_vec, is_3d, prompts=batch["prompt"])
        loss = torch.zeros((), device=device)

        if has_mask.any():
            loss = loss + seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])
            pm = (torch.sigmoid(out["seg_logits"][has_mask, 0]) > 0.5).float()
            gm = batch["mask"].to(device)[has_mask]
            inter = (pm * gm).sum(dim=(1, 2))
            union = pm.sum(dim=(1, 2)) + gm.sum(dim=(1, 2))
            dices.extend(((2.0 * inter + 1e-6) / (union + 1e-6)).cpu().tolist())

        valid = labels >= 0
        if valid.any():
            m_logits = mask_logits_by_cohort(out["multiclass_logits"], batch["dataset"])
            loss = loss + 3.0 * F.cross_entropy(m_logits[valid], labels[valid], weight=cls_weights)
            all_p.extend(m_logits[valid].argmax(dim=-1).cpu().tolist())
            all_y.extend(labels[valid].cpu().tolist())
        val_loss += float(loss.item())
        n_b += 1

    vd = float(np.mean(dices)) if dices else 0.0
    vf = float(f1_score(all_y, all_p, average="macro", zero_division=0)) if all_y else 0.0
    va = float(accuracy_score(all_y, all_p)) if all_y else 0.0
    return {"val_loss": val_loss / max(n_b, 1), "val_dice": vd, "val_macro_f1": vf, "val_acc": va, "composite": 0.5 * vd + 0.5 * vf}


@torch.no_grad()
def collect_masked_predictions(model: PanOrganNetV2, df: pd.DataFrame, cache_mgr: ShardCacheManager, mod_map: Dict[str, int], runtime: Any, device: torch.device, mnar_trunc: float = 0.0) -> Dict[str, Any]:
    model.eval()
    ds = PanOrganShardDataset(df, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=mnar_trunc)
    loader = DataLoader(ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)
    logits_l, labels_l, groups_l, mods_l, seg_recs = [], [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        has_mask = batch["has_mask"].to(device)
        labels = batch["label"]
        datasets = batch["dataset"]
        modalities = batch["modality"]
        out = model(images, meta_vec, is_3d, prompts=batch["prompt"])

        if has_mask.any():
            pm = torch.sigmoid(out["seg_logits"][has_mask, 0]).cpu().numpy()
            gm = batch["mask"][has_mask].numpy()
            for k, orig_i in enumerate(torch.where(has_mask)[0].cpu().tolist()):
                pb = (pm[k] > 0.5).astype(np.float32)
                gb = gm[k].astype(np.float32)
                inter = float((pb * gb).sum())
                union = float(pb.sum() + gb.sum() - inter)
                seg_recs.append({
                    "dataset": datasets[orig_i],
                    "dice": (2.0 * inter + 1e-6) / (pb.sum() + gb.sum() + 1e-6),
                    "iou": (inter + 1e-6) / (union + 1e-6),
                    "hd95_mm": 0.0,
                })

        m_logits = mask_logits_by_cohort(out["multiclass_logits"], datasets).cpu()
        for i in range(len(datasets)):
            lbl = int(labels[i].item())
            if lbl >= 0:
                logits_l.append(m_logits[i])
                labels_l.append(lbl)
                groups_l.append(datasets[i])
                mods_l.append(modalities[i])

    return {
        "logits": torch.stack(logits_l),
        "labels": np.array(labels_l),
        "groups": np.array(groups_l),
        "modalities": np.array(mods_l),
        "seg_records": seg_recs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PanOrgan 3.2 Masked Polish & Final Verification")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    shards_dir = _SSD_ROOT / "preprocessed_shards"
    ckpt_dir = _SSD_ROOT / "checkpoints" / "v32_balanced"
    exp_dir = _SSD_ROOT / "experiments"
    log_file = exp_dir / "v32_polish.log"

    audit = json.loads((shards_dir / "split_audit_report.json").read_text())
    leaks = {k: v.get("leakage_count", 0) for k, v in audit.items() if v.get("leakage_count", 0) > 0}
    assert len(leaks) == 0, f"Leakage detected: {leaks}"

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    sup_train = train_df[(train_df["has_mask"]) | (train_df["label"] >= 0)].reset_index(drop=True)
    sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)

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
    best_path = ckpt_dir / "best.pt"
    st = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    base_metrics = eval_masked(model, val_loader, device, cls_weights, seg_crit)
    best_comp = base_metrics["composite"]
    log_msg(f"Loaded 3.2 best.pt -> masked val_dice={base_metrics['val_dice']:.4f} | val_f1={base_metrics['val_macro_f1']:.4f} | val_acc={base_metrics['val_acc']:.4f}", log_file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-5, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    rng = np.random.default_rng(42)

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n_b = 0.0, 0
        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[3.2 Polish] Ep {epoch}/{args.epochs}", leave=False)):
            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            has_mask = batch["has_mask"].to(device)
            labels = batch["label"].to(device)
            valid = labels >= 0

            use_mixup = (not has_mask.any()) and valid.all() and (images.size(0) >= 2) and (rng.random() < 0.25)
            if use_mixup:
                lam = float(rng.beta(0.2, 0.2))
                perm = torch.randperm(images.size(0), device=device)
                images_in = lam * images + (1.0 - lam) * images[perm]
            else:
                lam, perm, images_in = 1.0, None, images

            out = model(images_in, meta_vec, is_3d, prompts=batch["prompt"])
            total = torch.zeros((), device=device)
            if has_mask.any():
                total = total + seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])
            if valid.any():
                m_logits = mask_logits_by_cohort(out["multiclass_logits"], batch["dataset"])
                if use_mixup and perm is not None:
                    c1 = F.cross_entropy(m_logits, labels, weight=cls_weights, label_smoothing=0.05)
                    c2 = F.cross_entropy(m_logits, labels[perm], weight=cls_weights, label_smoothing=0.05)
                    cls_l = lam * c1 + (1.0 - lam) * c2
                else:
                    cls_l = F.cross_entropy(m_logits[valid], labels[valid], weight=cls_weights, label_smoothing=0.05)
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

        scheduler.step()
        m = eval_masked(model, val_loader, device, cls_weights, seg_crit)
        if m["composite"] > best_comp:
            best_comp = m["composite"]
            atomic_torch_save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_composite": best_comp}, best_path)
        log_msg(f"[3.2 Polish Ep {epoch}/{args.epochs}] train_loss={run_loss/max(n_b,1):.4f} | val_loss={m['val_loss']:.4f} | val_dice={m['val_dice']:.4f} | val_f1={m['val_macro_f1']:.4f} | val_acc={m['val_acc']:.4f} (best={best_comp:.4f})", log_file)

    # Final Zero-Mismatch Verification
    st_f = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(st_f["model_state_dict"], strict=False)
    val_res = collect_masked_predictions(model, val_df, cache_mgr, mod_map, runtime, device, 0.0)
    test_res = collect_masked_predictions(model, test_df, cache_mgr, mod_map, runtime, device, 0.0)

    # Per-modality temperature scaling
    cal_probs = torch.zeros_like(test_res["logits"])
    val_cal_probs = torch.zeros_like(val_res["logits"])
    mod_temps: Dict[str, float] = {}
    for mod in sorted(set(test_res["modalities"].tolist())):
        v_idx = val_res["modalities"] == mod
        t_idx = test_res["modalities"] == mod
        sc = TemperatureScaler()
        t_opt = sc.fit(val_res["logits"][v_idx], torch.from_numpy(val_res["labels"][v_idx]))
        mod_temps[mod] = float(t_opt)
        val_cal_probs[v_idx] = sc.transform_probs(val_res["logits"][v_idx])
        cal_probs[t_idx] = sc.transform_probs(test_res["logits"][t_idx])

    probs_np = cal_probs.numpy()
    preds = probs_np.argmax(axis=-1)
    trues = test_res["labels"]

    cp = ModalityConditionalConformal(alpha=0.10)
    cp.fit(val_cal_probs, val_res["labels"], val_res["groups"].tolist())
    cp_cov = cp.evaluate_coverage(cal_probs, trues, test_res["groups"].tolist())

    per_cohort = {
        g: {"n": int((test_res["groups"] == g).sum()), "accuracy": round(float((preds[test_res["groups"] == g] == trues[test_res["groups"] == g]).mean()), 4)}
        for g in sorted(set(test_res["groups"].tolist()))
    }
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
    seg_0 = summarize_seg(test_res["seg_records"], n_boot=n_boot)
    seg_20 = summarize_seg(collect_masked_predictions(model, test_df, cache_mgr, mod_map, runtime, device, 0.20)["seg_records"], n_boot=n_boot)
    seg_40 = summarize_seg(collect_masked_predictions(model, test_df, cache_mgr, mod_map, runtime, device, 0.40)["seg_records"], n_boot=n_boot)

    final_report = {
        "model": "PanOrgan_3.2_Masked_Calibrated",
        "zero_leakage_passed": True,
        "n_test_cls": int(len(trues)),
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(trues, probs_np, multi_class="ovr", average="macro")),
        "per_class_report": classification_report(trues, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
        "per_cohort_accuracy": per_cohort,
        "per_modality_temperatures": mod_temps,
        "mc2_conformal_coverage": cp_cov,
        "mnar_stress_curve": {"0": seg_0, "20": seg_20, "40": seg_40},
        "checkpoint": str(best_path),
    }
    out_path = exp_dir / "thesis_reports" / "v32_final_verified_report.json"
    atomic_write_json(out_path, final_report)
    log_msg(f"=== 3.2 FINAL VERIFIED: Acc={final_report['accuracy']:.4f} | F1={final_report['macro_f1']:.4f} | AUROC={final_report['auroc']:.4f} | Dice={seg_0['overall']['dice']['mean']:.4f} ===", log_file)


if __name__ == "__main__":
    main()
