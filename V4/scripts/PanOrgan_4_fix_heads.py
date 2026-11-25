#!/usr/bin/env python3
"""PanOrgan 4 Multi-Scale Head & Calibration Fix:
Solves ct_binary task collision (kidney_ct_abn vs tcga_kirc_ct), global 8x8 pooling loss
on MG/CT/US-Nerve, Class-0 underfitting, high ECE (>0.05), and <90% MC2 conformal coverage
in ~10 minutes using cached multi-scale spatial features from PanOrgan 4 best.pt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_4"))
os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "V2"))
sys.path.insert(0, str(_REPO_ROOT / "V2" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V3" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V4" / "scripts"))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import PanOrganShardDataset, ShardCacheManager
from panorgan.evaluation import ModalityConditionalConformal, TemperatureScaler
from panorgan.models import PanOrganNetV2
from PanOrgan_4_optimize import PanOrgan4MultiHeadOpt
from PanOrgan_4_train import TASK_GROUP_MAP, TASK_NUM_CLASSES, _compute_ece, log_msg
from train_v3_pipeline import atomic_torch_save, atomic_write_json

# Dedicated cohort-level task routing to eliminate task collision (e.g. kidney_ct_abn vs tcga_kirc_ct)
COHORT_HEAD_MAP: Dict[str, Tuple[str, int]] = {
    "pulmonology_cxr": ("cxr_4class", 4),
    "pulmonology_cxr_ext": ("cxr_4class", 4),
    "nih_cxr14": ("cxr_4class", 4),
    "busi": ("busi_3class", 3),
    "cmmd_mg": ("cmmd_mg", 2),
    "cbis_ddsm": ("cbis_ddsm", 2),
    "kidney_ct_abn": ("kidney_ct_abn", 2),
    "tcga_kirc_ct": ("tcga_kirc_ct", 2),
    "tcga_lgg": ("tcga_lgg", 2),
    "prostatex_mri": ("prostatex_mri", 2),
    "ultrasound_nerve": ("ultrasound_nerve", 2),
}


def pool_multiscale(out: Dict[str, Any]) -> torch.Tensor:
    """Pool feats[1] (32x32), feats[2] (16x16), feats[3] (8x8) via Avg+Max + embedding -> 2688-d."""
    parts = [out["embedding"]]
    for f in out["features"][1:]:
        avg_p = f.mean(dim=(-2, -1))
        max_p = f.amax(dim=(-2, -1))
        parts.extend([avg_p, max_p])
    return torch.cat(parts, dim=-1)


class MultiScaleCohortHead(nn.Module):
    def __init__(self, in_dim: int = 2688, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def extract_split_features(
    model: PanOrgan4MultiHeadOpt,
    df: pd.DataFrame,
    cache_mgr: ShardCacheManager,
    mod_map: Dict[str, int],
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Dict[str, Any]:
    ds = PanOrganShardDataset(df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats_list, labels_list, cohorts_list = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        out1 = model.base(images, meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
        # Horizontal-flip TTA feature averaging for stability on MG/CT/US
        out2 = model.base(torch.flip(images, dims=[-1]), meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
        ms_feat = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
        feats_list.append(ms_feat.cpu())
        labels_list.append(batch["label"].cpu())
        cohorts_list.extend(batch["dataset"])
    return {
        "features": torch.cat(feats_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "cohorts": np.array(cohorts_list),
    }


def train_cohort_head(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    num_classes: int,
    epochs: int = 35,
) -> MultiScaleCohortHead:
    head = MultiScaleCohortHead(in_dim=x_tr.shape[1], num_classes=num_classes, dropout=0.35)
    counts = np.bincount(y_tr.numpy(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / counts) ** 0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32)

    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)

    best_score = -1.0
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    for _ in range(epochs):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = head(xb)
            ce = F.cross_entropy(logits, yb, weight=w, label_smoothing=0.03, reduction="none")
            pt = torch.exp(-F.cross_entropy(logits, yb, reduction="none")).clamp(1e-6, 1.0)
            loss = (((1.0 - pt) ** 1.5) * ce).mean()
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            v_logits = head(x_va)
            v_probs = F.softmax(v_logits, dim=-1).numpy()
            if len(np.unique(y_va.numpy())) > 1:
                auc = float(roc_auc_score(y_va.numpy(), v_probs[:, 1])) if num_classes == 2 else float(
                    roc_auc_score(y_va.numpy(), v_probs, multi_class="ovr", average="macro")
                )
            else:
                auc = 0.5
            f1 = float(f1_score(y_va.numpy(), v_logits.argmax(dim=-1).numpy(), average="macro", zero_division=0))
            score = 0.5 * auc + 0.5 * f1
            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    return head


def main():
    cfg = load_config(Path(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml"))
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = detect_runtime_profile(cfg)

    shards_dir = _V3_SSD / "preprocessed_shards"
    ckpt_dir = _SSD_ROOT / "checkpoints" / "panorgan_4"
    exp_dir = _SSD_ROOT / "experiments"
    log_file = exp_dir / "panorgan_4_pipeline.log"

    df = pd.read_parquet(shards_dir / "master_manifest_v4.parquet")
    df = df[df["label"] >= 0].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)

    base_model = PanOrganNetV2.from_config(cfg)
    model = PanOrgan4MultiHeadOpt(base_model, embed_dim=int(cfg["model"].get("embed_dim", 384)), dropout=0.30).to(device)
    st = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()
    log_msg("=== PanOrgan 4 Multi-Scale Head & Calibration Refinement Started ===", log_file)

    cache_file = exp_dir / "cached_multiscale_feats.pt"
    if cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        tr_data, va_data, te_data = cached["tr"], cached["va"], cached["te"]
        log_msg(f"Loaded cached multi-scale features from {cache_file}", log_file)
    else:
        tr_data = extract_split_features(model, train_df, cache_mgr, mod_map, runtime.batch_size, device, "Extract Train")
        va_data = extract_split_features(model, val_df, cache_mgr, mod_map, runtime.batch_size, device, "Extract Val")
        te_data = extract_split_features(model, test_df, cache_mgr, mod_map, runtime.batch_size, device, "Extract Test")
        atomic_torch_save({"tr": tr_data, "va": va_data, "te": te_data}, cache_file)
        log_msg(f"Saved cached multi-scale features to {cache_file}", log_file)

    # Train dedicated multi-scale heads per cohort/task-group
    head_specs = {}
    for ds, (hname, n_cls) in COHORT_HEAD_MAP.items():
        head_specs.setdefault(hname, {"num_classes": n_cls, "cohorts": []})["cohorts"].append(ds)

    trained_heads: Dict[str, MultiScaleCohortHead] = {}
    cohort_probs_val: Dict[str, np.ndarray] = {}
    cohort_probs_test: Dict[str, np.ndarray] = {}
    cohort_labels_val: Dict[str, np.ndarray] = {}
    cohort_labels_test: Dict[str, np.ndarray] = {}
    cohort_temps: Dict[str, float] = {}
    cohort_thrs: Dict[str, float] = {}

    for hname, info in head_specs.items():
        n_cls = info["num_classes"]
        cohorts = set(info["cohorts"])
        m_tr = np.isin(tr_data["cohorts"], list(cohorts)) & (tr_data["labels"].numpy() >= 0) & (tr_data["labels"].numpy() < n_cls)
        m_va = np.isin(va_data["cohorts"], list(cohorts)) & (va_data["labels"].numpy() >= 0) & (va_data["labels"].numpy() < n_cls)
        m_te = np.isin(te_data["cohorts"], list(cohorts)) & (te_data["labels"].numpy() >= 0) & (te_data["labels"].numpy() < n_cls)
        if m_tr.sum() == 0 or m_va.sum() == 0 or m_te.sum() == 0:
            log_msg(f"[PanOrgan 4 Head {hname:16s}] Skipped (train={m_tr.sum()}, val={m_va.sum()}, test={m_te.sum()})", log_file)
            continue

        head = train_cohort_head(
            tr_data["features"][m_tr], tr_data["labels"][m_tr],
            va_data["features"][m_va], va_data["labels"][m_va],
            num_classes=n_cls, epochs=40,
        )
        trained_heads[hname] = head

        with torch.no_grad():
            v_logits = head(va_data["features"][m_va])
            t_logits = head(te_data["features"][m_te])

        v_lbls = va_data["labels"][m_va].numpy()
        t_lbls = te_data["labels"][m_te].numpy()
        scaler = TemperatureScaler()
        t_opt = scaler.fit(v_logits, torch.from_numpy(v_lbls))
        v_probs = np.asarray(scaler.transform_probs(v_logits))
        t_probs = np.asarray(scaler.transform_probs(t_logits))

        best_thr = 0.50
        if n_cls == 2:
            best_f1 = -1.0
            for thr in np.linspace(0.25, 0.75, 51):
                vp = (v_probs[:, 1] >= thr).astype(int)
                f1_v = float(f1_score(v_lbls, vp, average="macro", zero_division=0))
                if f1_v > best_f1:
                    best_f1 = f1_v
                    best_thr = float(thr)

        cohort_probs_val[hname] = v_probs
        cohort_probs_test[hname] = t_probs
        cohort_labels_val[hname] = v_lbls
        cohort_labels_test[hname] = t_lbls
        cohort_temps[hname] = float(t_opt)
        cohort_thrs[hname] = float(best_thr)

        t_preds = (t_probs[:, 1] >= best_thr).astype(int) if n_cls == 2 else t_probs.argmax(axis=-1)
        auc = (float(roc_auc_score(t_lbls, t_probs[:, 1])) if n_cls == 2 else float(
            roc_auc_score(t_lbls, t_probs, multi_class="ovr", average="macro")
        )) if len(np.unique(t_lbls)) >= (2 if n_cls == 2 else n_cls) else 0.5
        ece = _compute_ece(t_probs, t_lbls)
        cp = ModalityConditionalConformal(alpha=0.10)
        cp.fit(v_probs, v_lbls, va_data["cohorts"][m_va].tolist())
        cp_cov = cp.evaluate_coverage(t_probs, t_lbls, te_data["cohorts"][m_te].tolist())
        log_msg(
            f"[PanOrgan 4 Head {hname:16s}] n={len(t_lbls):4d} | Acc={accuracy_score(t_lbls, t_preds):.4f} | "
            f"F1={f1_score(t_lbls, t_preds, average='macro', zero_division=0):.4f} | AUROC={auc:.4f} | ECE={ece:.4f} | MC2={cp_cov.get('overall', 0):.4f}",
            log_file,
        )

    # Save refined heads alongside PanOrgan 4 checkpoint
    atomic_torch_save({
        "multiscale_heads": {k: v.state_dict() for k, v in trained_heads.items()},
        "temperatures": cohort_temps,
        "thresholds": cohort_thrs,
    }, ckpt_dir / "multiscale_heads.pt")

    # Merge into panorgan_4_complete_thesis_report.json preserving existing segmentation/MNAR
    report_path = exp_dir / "thesis_reports" / "panorgan_4_complete_thesis_report.json"
    report = pd.read_json(report_path, typ="series").to_dict() if False else __import__("json").loads(report_path.read_text())

    all_preds, all_trues = [], []
    per_cohort_acc: Dict[str, Any] = {}
    per_head_report: Dict[str, Any] = {}
    for hname, info in head_specs.items():
        if hname not in trained_heads:
            continue
        n_cls = info["num_classes"]
        cohorts = set(info["cohorts"])
        m_va = np.isin(va_data["cohorts"], list(cohorts)) & (va_data["labels"].numpy() >= 0) & (va_data["labels"].numpy() < n_cls)
        m_te = np.isin(te_data["cohorts"], list(cohorts)) & (te_data["labels"].numpy() >= 0) & (te_data["labels"].numpy() < n_cls)
        v_probs = cohort_probs_val[hname]
        t_probs = cohort_probs_test[hname]
        v_lbls = cohort_labels_val[hname]
        t_lbls = cohort_labels_test[hname]
        thr = cohort_thrs[hname]
        t_preds = (t_probs[:, 1] >= thr).astype(int) if n_cls == 2 else t_probs.argmax(axis=-1)

        all_preds.extend(t_preds.tolist())
        all_trues.extend(t_lbls.tolist())

        te_c = te_data["cohorts"][m_te]
        for c in sorted(set(te_c.tolist())):
            idx = te_c == c
            per_cohort_acc[c] = {"n": int(idx.sum()), "accuracy": round(float((t_preds[idx] == t_lbls[idx]).mean()), 4)}

        cp = ModalityConditionalConformal(alpha=0.10)
        cp.fit(v_probs, v_lbls, va_data["cohorts"][m_va].tolist())
        cp_cov = cp.evaluate_coverage(t_probs, t_lbls, te_c.tolist())
        auc = (float(roc_auc_score(t_lbls, t_probs[:, 1])) if n_cls == 2 else float(
            roc_auc_score(t_lbls, t_probs, multi_class="ovr", average="macro")
        )) if len(np.unique(t_lbls)) >= (2 if n_cls == 2 else n_cls) else 0.5
        per_head_report[hname] = {
            "n": int(len(t_lbls)),
            "temperature": cohort_temps[hname],
            "decision_threshold": thr,
            "accuracy": float(accuracy_score(t_lbls, t_preds)),
            "macro_f1": float(f1_score(t_lbls, t_preds, average="macro", zero_division=0)),
            "auroc": auc,
            "ece": _compute_ece(t_probs, t_lbls),
            "confusion_matrix": confusion_matrix(t_lbls, t_preds).tolist(),
            "per_class_report": classification_report(t_lbls, t_preds, output_dict=True, zero_division=0),
            "mc2_conformal": cp_cov,
        }

    report["overall_classification"] = {
        "n_samples": int(len(all_trues)),
        "accuracy": float(accuracy_score(all_trues, all_preds)),
        "macro_f1": float(f1_score(all_trues, all_preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(all_trues, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_trues, all_preds, average="macro", zero_division=0)),
    }
    report["per_task_group"] = per_head_report
    report["per_cohort_accuracy"] = per_cohort_acc
    atomic_write_json(report_path, report)
    log_msg(
        f"=== PANORGAN 4 MULTISCALE REFINEMENT COMPLETED: Acc={report['overall_classification']['accuracy']:.4f} | Macro-F1={report['overall_classification']['macro_f1']:.4f} ===",
        log_file,
    )


if __name__ == "__main__":
    main()
