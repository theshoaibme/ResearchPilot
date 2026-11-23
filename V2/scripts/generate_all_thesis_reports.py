#!/usr/bin/env python3
"""Read-only Complete Thesis Report Generator for PanOrganNet-V2 & Ablations.

Reuses panorgan.evaluation helpers on untouched val/test splits without modifying checkpoints.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_SSD = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_V2"))
os.environ.setdefault("HF_HOME", str(_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_SSD / "model_cache" / "timm"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

REPO_V2 = Path(__file__).resolve().parent.parent
if str(REPO_V2) not in sys.path:
    sys.path.insert(0, str(REPO_V2))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import PanOrganShardDataset, ShardCacheManager
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


@torch.no_grad()
def run_inference(
    model: PanOrganNetV2,
    df_sub: pd.DataFrame,
    cache_mgr: ShardCacheManager,
    mod_map: Dict[str, int],
    runtime,
    mnar_trunc: float = 0.0,
) -> Dict[str, Any]:
    ds = PanOrganShardDataset(df_sub, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=mnar_trunc)
    loader = DataLoader(ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

    seg_records: Dict[str, List[Dict[str, float]]] = {}
    cls_logits, cls_labels, cls_groups = [], [], []

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
            ds_name = batch["dataset"][i]
            if has_mask[i]:
                pred_fg = preds_mask[i] > 0
                true_fg = true_mask[i] > 0
                inter = np.logical_and(pred_fg, true_fg).sum()
                union = np.logical_or(pred_fg, true_fg).sum()
                iou = float(inter / union) if union > 0 else (1.0 if not true_fg.any() else 0.0)
                res = compute_dice_and_hd95(pred_fg, true_fg, spacing_mm=(1.5, 1.5))
                res["iou"] = iou
                seg_records.setdefault(ds_name, []).append(res)

            lbl = int(batch["label"][i].item())
            if lbl >= 0:
                cls_logits.append(out["multiclass_logits"][i].cpu())
                cls_labels.append(lbl)
                cls_groups.append(ds_name)

    return {
        "seg_records": seg_records,
        "cls_logits": torch.stack(cls_logits) if cls_logits else torch.empty(0, 4),
        "cls_labels": np.array(cls_labels, dtype=np.int64),
        "cls_groups": cls_groups,
    }


def summarize_seg(seg_records: Dict[str, List[Dict[str, float]]], n_boot: int) -> Dict[str, Any]:
    out = {}
    all_dice, all_iou, all_hd95 = [], [], []
    for ds_name, items in seg_records.items():
        dices = np.array([x["dice"] for x in items])
        ious = np.array([x["iou"] for x in items])
        hd95s = np.array([x["hd95_mm"] for x in items])
        all_dice.extend(dices.tolist())
        all_iou.extend(ious.tolist())
        all_hd95.extend(hd95s.tolist())
        out[ds_name] = {
            "n_samples": len(items),
            "dice": bootstrap_confidence_interval(dices, n_bootstrap=n_boot),
            "iou": bootstrap_confidence_interval(ious, n_bootstrap=n_boot),
            "hd95_mm": bootstrap_confidence_interval(hd95s, n_bootstrap=n_boot),
        }
    if all_dice:
        out["overall"] = {
            "n_samples": len(all_dice),
            "dice": bootstrap_confidence_interval(np.array(all_dice), n_bootstrap=n_boot),
            "iou": bootstrap_confidence_interval(np.array(all_iou), n_bootstrap=n_boot),
            "hd95_mm": bootstrap_confidence_interval(np.array(all_hd95), n_bootstrap=n_boot),
        }
    return out


def compute_cls_full(logits: torch.Tensor, labels: np.ndarray, scaler: TemperatureScaler | None = None) -> Dict[str, Any]:
    if len(labels) == 0:
        return {}
    probs = scaler.transform_probs(logits) if scaler is not None else torch.softmax(logits, dim=-1).numpy()
    # Restrict to observed active classes (0..max_lbl) for clean multiclass metrics
    n_active = int(labels.max()) + 1
    probs_act = probs[:, :n_active]
    probs_act = probs_act / np.clip(probs_act.sum(axis=1, keepdims=True), 1e-12, None)
    preds = probs_act.argmax(axis=1)

    base = compute_classification_metrics(labels, probs_act)
    # Macro Sensitivity (Recall) & Specificity
    cm = confusion_matrix(labels, preds, labels=list(range(n_active)))
    sens_list, spec_list, prec_list = [], [], []
    for c in range(n_active):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens_list.append(tp / max(tp + fn, 1))
        spec_list.append(tn / max(tn + fp, 1))
        prec_list.append(tp / max(tp + fp, 1))

    cal = compute_ece_and_brier(labels, probs_act, n_bins=15)
    return {
        **base,
        "precision_macro": float(np.mean(prec_list)),
        "recall_sensitivity_macro": float(np.mean(sens_list)),
        "specificity_macro": float(np.mean(spec_list)),
        "confusion_matrix": cm.tolist(),
        "ece": float(cal["ece"]),
        "brier": float(cal["brier"]),
        "n_samples": int(len(labels)),
    }


def main() -> None:
    cfg = load_config(REPO_V2 / "configs" / "v2_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)

    shards_dir = _SSD / "preprocessed_shards"
    ckpt_dir = _SSD / "checkpoints"
    out_dir = _SSD / "experiments" / "thesis_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    n_boot = int(cfg["evaluation"]["bootstrap_iterations"])

    models_spec = {
        "PanOrganNet_V2_Full": ("stage2_panorgan_v2_full", {}),
        "Ablation_A1_RandomMask": ("ablation_A1_random_mask", {"masking_mode": "random", "use_moe": True, "use_sacr": True}),
        "Ablation_A2_GradSGM": ("ablation_A2_grad_sgm", {"masking_mode": "gradient_sgm", "use_moe": True, "use_sacr": True}),
        "Ablation_A3_NoMoE": ("ablation_A3_no_moe", {"masking_mode": "ma_ugm", "use_moe": False, "use_sacr": True}),
        "Ablation_A4_NoSACR": ("ablation_A4_no_sacr", {"masking_mode": "ma_ugm", "use_moe": True, "use_sacr": False}),
    }

    all_results: Dict[str, Any] = {}
    main_val_inf = None
    main_test_inf = None

    for name, (ckpt_sub, overrides) in models_spec.items():
        ckpt_path = ckpt_dir / ckpt_sub / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] Missing {ckpt_path}")
            continue
        print(f"Evaluating {name} ({ckpt_path})...", flush=True)
        m_cfg = copy.deepcopy(cfg)
        m_cfg["model"].update(overrides)
        model = PanOrganNetV2.from_config(m_cfg).to(runtime.device)
        st = torch.load(ckpt_path, map_location=runtime.device, weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=False)
        model.eval()

        test_inf = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
        seg_sum = summarize_seg(test_inf["seg_records"], n_boot=n_boot)

        if name == "PanOrganNet_V2_Full":
            val_inf = run_inference(model, val_df, cache_mgr, mod_map, runtime, mnar_trunc=0.0)
            main_val_inf = val_inf
            main_test_inf = test_inf

            # Temperature Scaling + MC2-CP Conformal Prediction
            scaler = TemperatureScaler()
            if len(val_inf["cls_labels"]) > 0:
                t_opt = scaler.fit(val_inf["cls_logits"], torch.from_numpy(val_inf["cls_labels"]))
            else:
                t_opt = 1.0

            cls_raw = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=None)
            cls_cal = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=scaler)

            # MC2-CP Conformal Prediction (target coverage 90% -> alpha=0.10)
            cp_cov = {}
            if len(val_inf["cls_labels"]) > 0 and len(test_inf["cls_labels"]) > 0:
                cp = ModalityConditionalConformal(alpha=0.10)
                val_probs = scaler.transform_probs(val_inf["cls_logits"])
                test_probs = scaler.transform_probs(test_inf["cls_logits"])
                cp.fit(val_probs, val_inf["cls_labels"], val_inf["cls_groups"])
                cp_cov = cp.evaluate_coverage(test_probs, test_inf["cls_labels"], test_inf["cls_groups"])

            # MNAR Stress Curve (0%, 20%, 40%)
            mnar_curve = {"0": seg_sum}
            for frac in (0.20, 0.40):
                inf_f = run_inference(model, test_df, cache_mgr, mod_map, runtime, mnar_trunc=frac)
                mnar_curve[str(int(frac * 100))] = summarize_seg(inf_f["seg_records"], n_boot=n_boot)

            all_results[name] = {
                "segmentation": seg_sum,
                "classification_uncalibrated": cls_raw,
                "classification_calibrated": cls_cal,
                "optimal_temperature": float(t_opt),
                "mc2_conformal_coverage": cp_cov,
                "mnar_stress_curve": mnar_curve,
            }
        else:
            cls_sum = compute_cls_full(test_inf["cls_logits"], test_inf["cls_labels"], scaler=None)
            all_results[name] = {
                "segmentation": seg_sum,
                "classification": cls_sum,
            }

    # Save complete JSON report
    report_path = out_dir / "complete_thesis_report.json"
    report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Export LaTeX table for Segmentation & Ablations
    rows = []
    for name, res in all_results.items():
        seg = res["segmentation"]
        cls_d = res.get("classification_calibrated") or res.get("classification", {})
        rows.append({
            "Model": name.replace("_", " "),
            "MRI_Dice": f"{seg['tcga_lgg']['dice']['mean']:.4f}" if "tcga_lgg" in seg else "-",
            "US_Dice": f"{seg['busi']['dice']['mean']:.4f}" if "busi" in seg else "-",
            "Overall_Dice": f"{seg['overall']['dice']['mean']:.4f}" if "overall" in seg else "-",
            "Acc": f"{cls_d.get('accuracy', 0.0):.4f}",
            "F1": f"{cls_d.get('f1_macro', 0.0):.4f}",
            "AUC": f"{cls_d.get('auroc_macro', 0.0):.4f}",
        })
    cols = ["Model", "MRI_Dice", "US_Dice", "Overall_Dice", "Acc", "F1", "AUC"]
    tex = export_latex_results_table(rows, columns=cols, caption="PanOrganNet-V2 & Ablations Held-Out Test Comparison", label="tab:panorgan_v2_results")
    (out_dir / "table_panorgan_v2_results.tex").write_text(tex, encoding="utf-8")

    # Plot 1: Confusion Matrix & Calibration / MNAR plots for Main Model
    if main_test_inf is not None and "PanOrganNet_V2_Full" in all_results:
        res_main = all_results["PanOrganNet_V2_Full"]
        cm = np.array(res_main["classification_calibrated"]["confusion_matrix"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        im = ax1.imshow(cm, cmap="Blues")
        ax1.set_title("Confusion Matrix (Held-Out Test)")
        ax1.set_xlabel("Predicted Class")
        ax1.set_ylabel("True Class")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax1.text(c, r, str(cm[r, c]), ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")

        # MNAR Stress Curve
        truncs = [0, 20, 40]
        mri_d = [res_main["mnar_stress_curve"][str(t)]["tcga_lgg"]["dice"]["mean"] for t in truncs]
        us_d = [res_main["mnar_stress_curve"][str(t)]["busi"]["dice"]["mean"] for t in truncs]
        ax2.plot(truncs, mri_d, "o-", lw=2.2, label="Brain MRI (TCGA-LGG)")
        ax2.plot(truncs, us_d, "s-", lw=2.2, label="Breast US (BUSI)")
        ax2.set_title("MNAR Partial-FOV Stress Curve")
        ax2.set_xlabel("FOV Truncation (%)")
        ax2.set_ylabel("Dice Score")
        ax2.set_ylim(0.4, 0.95)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "thesis_evaluation_plots.png")
        plt.close(fig)

    print(f"ALL THESIS REPORTS GENERATED -> {out_dir}")


if __name__ == "__main__":
    main()
