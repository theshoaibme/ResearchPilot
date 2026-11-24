#!/usr/bin/env python3
"""Comprehensive Pre-Deployment Validation Gate for PanOrgan 3.2."""
from __future__ import annotations

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
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "V2"))
sys.path.insert(0, str(REPO_ROOT / "V2/scripts"))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import ShardCacheManager
from panorgan.evaluation import ModalityConditionalConformal, TemperatureScaler
from panorgan.models import PanOrganNetV2
from generate_all_thesis_reports import compute_cls_full, run_inference, summarize_seg


def main() -> None:
    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    shards_dir = _SSD_ROOT / "preprocessed_shards"
    ckpt_path = _SSD_ROOT / "checkpoints" / "v32_balanced" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not ready yet: {ckpt_path}")

    # 1. Split Audit Check
    audit = json.loads((shards_dir / "split_audit_report.json").read_text())
    leaks = {k: v.get("leakage_count", 0) for k, v in audit.items() if v.get("leakage_count", 0) > 0}

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)

    model = PanOrganNetV2.from_config(cfg).to(device)
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()

    val_inf = run_inference(model, val_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
    test_inf = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)

    # 2. Overall & Per-Class Classification Report
    logits = test_inf["cls_logits"]
    labels = test_inf["cls_labels"]
    groups = np.array(test_inf["cls_groups"])
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    per_class = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    # 3. Per-Cohort Classification Breakdown
    per_cohort_cls: Dict[str, Any] = {}
    for g in sorted(set(groups.tolist())):
        idx = groups == g
        g_acc = float((preds[idx] == labels[idx]).mean())
        per_cohort_cls[g] = {"n": int(idx.sum()), "accuracy": round(g_acc, 4)}

    # 4. Calibration & Conformal
    scaler = TemperatureScaler()
    t_opt = scaler.fit(val_inf["cls_logits"], torch.from_numpy(val_inf["cls_labels"]))
    cls_raw = compute_cls_full(logits, labels, scaler=None)
    cls_cal = compute_cls_full(logits, labels, scaler=scaler)

    cp = ModalityConditionalConformal(alpha=0.10)
    cp.fit(scaler.transform_probs(val_inf["cls_logits"]), val_inf["cls_labels"], val_inf["cls_groups"])
    cp_cov = cp.evaluate_coverage(scaler.transform_probs(logits), labels, test_inf["cls_groups"])

    # 5. Segmentation + MNAR Stress Curve (0%, 20%, 40%)
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])
    seg_0 = summarize_seg(test_inf["seg_records"], n_boot=n_boot)
    seg_20 = summarize_seg(run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.20)["seg_records"], n_boot=n_boot)
    seg_40 = summarize_seg(run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.40)["seg_records"], n_boot=n_boot)

    # 6. Clinical Case Spot-Checks (2 per cohort)
    spot_checks: List[Dict[str, Any]] = []
    for g in sorted(set(groups.tolist())):
        idxs = np.where(groups == g)[0][:2]
        for i in idxs:
            spot_checks.append({
                "cohort": g,
                "true_label": int(labels[i]),
                "pred_label": int(preds[i]),
                "confidence": round(float(probs[i, preds[i]]), 4),
                "correct": bool(labels[i] == preds[i]),
            })

    report = {
        "checkpoint": str(ckpt_path),
        "zero_leakage_passed": len(leaks) == 0,
        "leaks": leaks,
        "per_class_report": per_class,
        "confusion_matrix": cm,
        "per_cohort_classification": per_cohort_cls,
        "classification_uncalibrated": cls_raw,
        "classification_calibrated": cls_cal,
        "optimal_temperature": float(t_opt),
        "mc2_conformal_coverage": cp_cov,
        "mnar_stress_curve": {"0": seg_0, "20": seg_20, "40": seg_40},
        "clinical_spot_checks": spot_checks,
    }

    out_path = _SSD_ROOT / "experiments" / "thesis_reports" / "v32_predeployment_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PRE-DEPLOYMENT VALIDATION SAVED -> {out_path}")
    print(f"Zero-Leakage: {report['zero_leakage_passed']} | Acc: {cls_cal['accuracy']:.4f} | Macro-F1: {cls_cal['macro_f1']:.4f} | Overall Dice: {seg_0['overall']['dice']['mean']:.4f}")


if __name__ == "__main__":
    main()
