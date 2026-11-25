#!/usr/bin/env python3
"""Generate:
1. Expanded 20-sample (4x5) v4_sample_predictions.png across all 10 supervised test cohorts
   using real inference (best.pt + multiscale_heads.pt for strong cohorts, weak_cohort_specialist.pt for weak cohorts).
2. Clean, full-scale v4_train_history.png (Base Ep 1-6 + Opt Ep 7-9 + Weak-FT Ep 1-4, no misleading zoomed-in cliffs).
3. Sync all latest plots + thesis_report.json + README.md to PanOrgan-4-CardUpdate and upload to Hugging Face (theshoaibme/PanOrgan-4).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_SSD_ROOT = Path("/Volumes/Shoaib/PanOrgan_4")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HF_STAGE = Path("/Volumes/Shoaib/PanOrgan_HF_Staging/PanOrgan-4-CardUpdate")

os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

sys.path.insert(0, str(_REPO_ROOT / "V2"))
sys.path.insert(0, str(_REPO_ROOT / "V2" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V3" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V4" / "scripts"))

from panorgan.config import load_config
from panorgan.data import PanOrganShardDataset, ShardCacheManager
from panorgan.models import PanOrganNetV2
from PanOrgan_4_fix_heads import COHORT_HEAD_MAP, MultiScaleCohortHead, pool_multiscale
from PanOrgan_4_finetune_weak import WEAK_COHORTS, WeakCohortEndToEndNet

STRONG_COHORTS = ["pulmonology_cxr", "pulmonology_cxr_ext", "nih_cxr14", "busi", "tcga_lgg"]
ALL_10_COHORTS = STRONG_COHORTS + WEAK_COHORTS

CLASS_NAMES: Dict[str, List[str]] = {
    "cxr_4class": ["Normal", "Pneumonia", "COVID-19", "Effusion"],
    "busi_3class": ["Normal", "Benign", "Malignant"],
    "tcga_lgg": ["Grade II", "Grade III"],
    "cmmd_mg": ["Benign", "Malignant"],
    "cbis_ddsm": ["Benign", "Malignant"],
    "kidney_ct_abn": ["Normal", "Abnormal"],
    "tcga_kirc_ct": ["Low Grade", "High Grade"],
    "ultrasound_nerve": ["No Nerve", "BP Nerve"],
}

COHORT_TITLES: Dict[str, str] = {
    "pulmonology_cxr": "CXR (Pulm-1)",
    "pulmonology_cxr_ext": "CXR (Pulm-Ext)",
    "nih_cxr14": "CXR (NIH-14)",
    "busi": "Breast US (BUSI)",
    "tcga_lgg": "Brain MRI (LGG)",
    "cmmd_mg": "Mammo (CMMD)",
    "cbis_ddsm": "Mammo (CBIS)",
    "kidney_ct_abn": "Kidney CT (Abn)",
    "tcga_kirc_ct": "Kidney CT (KIRC)",
    "ultrasound_nerve": "Nerve US (BP)",
}


def plot_clean_train_history(out_paths: List[Path]) -> None:
    epochs_main = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_loss = [1.9333, 1.7557, 1.7087, 1.6420, 1.5949, 1.5570, 1.0717, 1.0642, 1.0234]
    val_dice = [0.0752, 0.8677, 0.8737, 0.8745, 0.8691, 0.8739, 0.8769, 0.8806, 0.8813]
    val_f1 = [0.8660, 0.8535, 0.8598, 0.8546, 0.8652, 0.8686, 0.8631, 0.8619, 0.8610]
    val_acc = [0.8150, 0.7993, 0.8081, 0.7976, 0.8104, 0.8154, 0.8085, 0.8075, 0.8062]

    w_ep = [1, 2, 3, 4]
    w_loss = [0.0834, 0.0772, 0.0538, 0.0403]
    w_acc = [0.6715, 0.6443, 0.6900, 0.6780]
    w_f1 = [0.6396, 0.6287, 0.6615, 0.6524]
    w_auc = [0.6939, 0.7107, 0.7128, 0.7112]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("PanOrgan 4 — Complete Training & Calibration Trajectory (Base Ep 1-6 + Opt Ep 7-9 + Weak-FT)", fontsize=13, fontweight="bold", y=0.98)

    # Panel 1: Training Loss (Base + Opt)
    ax = axes[0, 0]
    ax.plot(epochs_main, train_loss, "o-", color="#2563eb", lw=2.2, label="Joint Multi-Task Train Loss")
    ax.axvline(6.5, color="#64748b", ls="--", lw=1.2, label="Opt Phase Start (Focal + Calib)")
    ax.set_title("Training Loss Trajectory (Epochs 1-9)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.8, 2.1)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Validation Accuracy & Macro-F1 (Proper full scale [0.70, 0.92])
    ax = axes[0, 1]
    ax.plot(epochs_main, [a * 100 for a in val_acc], "s-", color="#059669", lw=2.2, label="Val Weighted Accuracy (%)")
    ax.plot(epochs_main, [f * 100 for f in val_f1], "o-", color="#7c3aed", lw=2.2, label="Val Macro-F1 (%)")
    ax.axvline(6.5, color="#64748b", ls="--", lw=1.2)
    ax.set_title("Validation Classification Stability (Full Scale)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70.0, 92.0)
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel 3: Validation Segmentation Dice
    ax = axes[1, 0]
    ax.plot(epochs_main, [d * 100 for d in val_dice], "d-", color="#0284c7", lw=2.2, label="Val Segmentation Dice (%)")
    ax.axvline(6.5, color="#64748b", ls="--", lw=1.2)
    ax.set_title("Universal Segmentation Dice Convergence", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(0.0, 96.0)
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel 4: Weak-Cohort End-to-End Fine-Tuning (Ep 1-4)
    ax = axes[1, 1]
    ax.plot(w_ep, [a * 100 for a in w_acc], "o-", color="#ea580c", lw=2.2, label="Weak Val Acc (%)")
    ax.plot(w_ep, [f * 100 for f in w_f1], "s-", color="#d97706", lw=2.2, label="Weak Val F1 (%)")
    ax.plot(w_ep, [u * 100 for u in w_auc], "^-", color="#0d9488", lw=2.2, label="Weak Val AUROC (%)")
    ax.set_title("Weak-Cohort End-to-End Fine-Tuning (Ep 1-4)", fontweight="bold")
    ax.set_xlabel("Weak-FT Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(w_ep)
    ax.set_ylim(55.0, 80.0)
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_20_sample_predictions(out_paths: List[Path]) -> None:
    device = torch.device("cpu")
    cfg = load_config(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml")
    shards_dir = _V3_SSD / "preprocessed_shards"
    ckpt_dir = _SSD_ROOT / "checkpoints" / "panorgan_4"

    df = pd.read_parquet(shards_dir / "master_manifest_v4.parquet")
    test_df = df[(df["split"] == "test") & (df["dataset"].isin(ALL_10_COHORTS)) & (df["label"] >= 0)].reset_index(drop=True)

    # Pick 2 representative non-blank samples per cohort (1 class 0, 1 class >=1 if available)
    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=50.0)
    full_test_ds = PanOrganShardDataset(test_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

    selected_indices = []
    for ds in ALL_10_COHORTS:
        sub_idx = test_df.index[test_df["dataset"] == ds].tolist()
        neg_pick, pos_pick = None, None
        for i in sub_idx:
            lbl = int(test_df.loc[i, "label"])
            if (lbl == 0 and neg_pick is not None) or (lbl > 0 and pos_pick is not None):
                continue
            item = full_test_ds[i]
            raw = item["image"].numpy()
            slc = raw[2] if bool(item["is_3d"].item()) else raw[0]
            if float(slc.std()) < 0.05:
                continue
            if lbl == 0 and neg_pick is None:
                neg_pick = i
            elif lbl > 0 and pos_pick is None:
                pos_pick = i
            if neg_pick is not None and pos_pick is not None:
                break
        picks = [p for p in [neg_pick, pos_pick] if p is not None]
        if len(picks) < 2:
            for i in sub_idx:
                if i not in picks:
                    picks.append(i)
                if len(picks) == 2:
                    break
        selected_indices.extend(picks[:2])

    sel_df = test_df.loc[selected_indices].reset_index(drop=True)
    sel_ds = PanOrganShardDataset(sel_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

    # Load Strong model (best.pt + multiscale_heads.pt)
    strong_base = PanOrganNetV2.from_config(cfg)
    st_best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    strong_base.load_state_dict({k[len("base."):]: v for k, v in st_best["model_state_dict"].items() if k.startswith("base.")}, strict=False)
    strong_base.eval()

    ms_ck = torch.load(ckpt_dir / "multiscale_heads.pt", map_location="cpu", weights_only=False)
    strong_heads: Dict[str, MultiScaleCohortHead] = {}
    for hname, n_cls in set(COHORT_HEAD_MAP.values()):
        if hname in ms_ck["multiscale_heads"]:
            h = MultiScaleCohortHead(2688, 512, n_cls, dropout=0.35)
            h.load_state_dict(ms_ck["multiscale_heads"][hname])
            h.eval()
            strong_heads[hname] = h

    # Load Weak specialist model
    weak_base = PanOrganNetV2.from_config(cfg)
    weak_heads = {ds: MultiScaleCohortHead(2688, 512, 2, dropout=0.30) for ds in WEAK_COHORTS}
    weak_net = WeakCohortEndToEndNet(weak_base, weak_heads)
    st_weak = torch.load(ckpt_dir / "weak_cohort_specialist.pt", map_location="cpu", weights_only=False)
    weak_net.load_state_dict(st_weak["model_state_dict"], strict=False)
    weak_net.eval()

    # Run inference on 20 samples
    fig, axes = plt.subplots(4, 5, figsize=(18, 14.5), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "PanOrgan 4 — Representative Test Set Predictions Across All 10 Clinical Cohorts (n=20)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    for idx in range(len(sel_ds)):
        row_idx, col_idx = divmod(idx, 5)
        ax = axes[row_idx, col_idx]
        item = sel_ds[idx]
        ds = str(item["dataset"])
        true_lbl = int(item["label"].item())
        img_t = item["image"].unsqueeze(0).to(device)
        meta_t = item["meta_vec"].unsqueeze(0).to(device)
        is_3d_t = item["is_3d"].unsqueeze(0).to(device)
        prompt = [str(item["prompt"])]

        if ds in WEAK_COHORTS:
            out1 = weak_net.base(img_t, meta_t, is_3d_t, prompts=prompt, return_features=True)
            out2 = weak_net.base(torch.flip(img_t, dims=[-1]), meta_t, is_3d_t, prompts=prompt, return_features=True)
            ms_feat = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
            logits = weak_net.heads[ds](ms_feat)[0]
            hname = ds
        else:
            hname, _ = COHORT_HEAD_MAP[ds]
            out1 = strong_base(img_t, meta_t, is_3d_t, prompts=prompt, return_features=True)
            out2 = strong_base(torch.flip(img_t, dims=[-1]), meta_t, is_3d_t, prompts=prompt, return_features=True)
            ms_feat = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
            logits = strong_heads[hname](ms_feat)[0]

        probs = F.softmax(logits, dim=-1).numpy()
        pred_lbl = int(probs.argmax())
        conf = float(probs[pred_lbl]) * 100.0

        # Extract display grayscale slice
        raw = item["image"].numpy()
        slc = raw[2] if bool(item["is_3d"].item()) else raw[0]
        slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-6)

        ax.imshow(slc, cmap="gray")
        # Overlay predicted segmentation contour if has_mask
        if bool(item.get("has_mask", False)):
            seg_prob = torch.sigmoid(out1["seg_logits"][0, 0]).numpy()
            if seg_prob.max() > 0.35:
                ax.contour(seg_prob, levels=[0.5], colors=["#22c55e"], linewidths=1.2)

        cnames = CLASS_NAMES.get(hname, ["Neg", "Pos"])
        t_str = cnames[true_lbl] if true_lbl < len(cnames) else str(true_lbl)
        p_str = cnames[pred_lbl] if pred_lbl < len(cnames) else str(pred_lbl)
        ok = (true_lbl == pred_lbl)
        color = "#15803d" if ok else "#b91c1c"
        mark = "✓" if ok else "✗"

        ax.set_title(
            f"{COHORT_TITLES.get(ds, ds)}\nTrue: {t_str} | Pred: {p_str} ({conf:.1f}%) {mark}",
            fontsize=9.5,
            fontweight="bold",
            color=color,
            pad=5,
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)


def update_hf_card_and_push() -> None:
    report_src = _SSD_ROOT / "experiments" / "thesis_reports" / "panorgan_4_complete_thesis_report.json"
    shutil.copy2(report_src, _HF_STAGE / "thesis_report.json")

    # Copy all latest plots from reports/assets to HF figures/
    assets_dir = _REPO_ROOT / "reports" / "assets"
    hf_figs = _HF_STAGE / "figures"
    hf_figs.mkdir(parents=True, exist_ok=True)
    for fn in [
        "v4_train_history.png",
        "v4_per_head_metrics.png",
        "v4_cohort_accuracy.png",
        "v4_calibration_ece.png",
        "v4_confusion_matrices.png",
        "v4_mnar_stress.png",
        "v4_data_distribution.png",
        "v4_sample_predictions.png",
    ]:
        src = assets_dir / fn
        if src.exists():
            shutil.copy2(src, hf_figs / fn)

    with open(report_src, "r", encoding="utf-8") as f:
        rep = json.load(f)

    oc = rep["overall_classification"]
    ptg = rep["per_task_group"]
    seg = rep["segmentation"]["overall"]
    mnar = rep["mnar_stress_curve"]

    readme = f"""---
license: apache-2.0
tags:
- medical-imaging
- multimodal
- radiology
- foundation-model
- pytorch
- convnext
- segmentation
- classification
datasets:
- theshoaibme/PanOrgan-Dataset
metrics:
- accuracy
- f1
- roc_auc
---

# PanOrgan 4 — Unified 5-Modality Clinical Foundation Model

**PanOrgan 4** is a unified 2D + 2.5D multimodal medical imaging foundation model trained across **5 modalities** (`CXR`, `CT`, `MRI`, `US`, `MG`) and **11 clinical cohorts** (`20,175` sharded samples from `9,381` patients with strict patient-level zero-leakage splitting).

It pairs a shared `PanOrganNetV2` backbone (`OmniModalityStem` + `PhysicsFiLM` + `ConvNeXt-Small` + `LoRA` + `Sparse Top-2 MoE` + `SACR`) with **cohort-isolated multi-scale spatial heads** (`32×32`, `16×16`, `8×8` Avg+Max pooled + `384-d` embedding = `2,688-d` with flip TTA), per-cohort Class-Balanced Focal Loss, Temperature Scaling, decision threshold tuning, and end-to-end weak-cohort fine-tuning (`Weak-FT`).

---

## Key Test Results (`n = {oc['n_samples']}` Classification, `n = {seg['n_samples']}` Segmentation)

- **Overall Weighted Accuracy:** **{oc['accuracy']*100:.2f}%**
- **Mean Head Macro-F1:** **{oc['macro_f1']*100:.2f}%**
- **Segmentation Dice (Clean):** **{seg['dice']['mean']:.4f}** (IoU `{seg['iou']['mean']:.4f}`)
- **MNAR FOV Occlusion Robustness:** Dice `{mnar['0']['overall']['dice']['mean']:.4f}` (0%) → `{mnar['20']['overall']['dice']['mean']:.4f}` (20% FOV loss) → `{mnar['40']['overall']['dice']['mean']:.4f}` (40% FOV loss)

### Per-Head Test Performance

| Clinical Head | Samples | Accuracy | Macro-F1 | AUROC | ECE |
|---|---:|---:|---:|---:|---:|
"""
    for hname, m in ptg.items():
        readme += f"| `{hname}` | {m['n']} | {m['accuracy']*100:.2f}% | {m['macro_f1']*100:.2f}% | {m['auroc']:.4f} | {m['ece']:.4f} |\n"

    readme += """
---

## Visualizations

### 1. Representative Test Set Predictions (All 10 Supervised Cohorts, `n = 20`)
![Sample Predictions](figures/v4_sample_predictions.png)

### 2. Training & Calibration Trajectory
![Training History](figures/v4_train_history.png)

### 3. Per-Head Metrics & Cohort Accuracy
![Per-Head Metrics](figures/v4_per_head_metrics.png)
![Cohort Accuracy](figures/v4_cohort_accuracy.png)

### 4. Calibration (ECE) & Confusion Matrices
![Calibration ECE](figures/v4_calibration_ece.png)
![Confusion Matrices](figures/v4_confusion_matrices.png)

### 5. MNAR Field-of-View Occlusion Robustness
![MNAR Stress](figures/v4_mnar_stress.png)
"""
    (_HF_STAGE / "README.md").write_text(readme, encoding="utf-8")

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(_HF_STAGE),
        repo_id="theshoaibme/PanOrgan-4",
        repo_type="model",
        commit_message="Update PanOrgan 4 with Weak-FT metrics, 20-sample prediction grid, and full-scale training curves",
    )
    print("HF upload complete!")


def main() -> None:
    plot_clean_train_history([
        _REPO_ROOT / "reports" / "assets" / "v4_train_history.png",
        _REPO_ROOT / "web" / "static" / "plots" / "v4_train_history.png",
    ])
    print("Generated clean v4_train_history.png")

    generate_20_sample_predictions([
        _REPO_ROOT / "reports" / "assets" / "v4_sample_predictions.png",
        _REPO_ROOT / "web" / "static" / "plots" / "v4_sample_predictions.png",
    ])
    print("Generated 20-sample v4_sample_predictions.png")

    update_hf_card_and_push()


if __name__ == "__main__":
    main()
