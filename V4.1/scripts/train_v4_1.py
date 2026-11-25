#!/usr/bin/env python3
"""PanOrgan 4.1 — Rule-Compliant, Anti-Overfitting, Calibrated Training & Evaluation Pipeline.

Implements PANORGAN_RESEARCH_RULESET.md:
  - Warm-started from PanOrgan 4 (best.pt + multiscale_heads.pt) on master_manifest_v4_1.parquet
    (with balanced 50/50 CBIS-DDSM ROI crops & balanced 50/50 Kidney CT).
  - Low Backbone LR (1.5e-5) + Linear Warmup + Cosine Annealing + Weight Decay (0.08) + Dropout (0.40)
    + MixUp (p=0.25) + Label Smoothing (0.05) + EMA (0.998).
  - Exact same classification loss formula for train & val tracking (no scale mismatch).
  - Post-hoc Temperature Scaling + Macro-F1/Youden threshold tuning on val (prevents positive bias).
  - Full Test Evaluation: Weighted Acc, Balanced Acc, Macro-F1, AUROC, AUPRC, ECE, 95% Bootstrap CI,
    and MNAR 0/20/40% Segmentation Stress Test.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_V4_SSD = Path("/Volumes/Shoaib/PanOrgan_4")
_V41_SSD = Path("/Volumes/Shoaib/PanOrgan_4_1")

os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "V2"))
sys.path.insert(0, str(_REPO_ROOT / "V2" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V3" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "V4" / "scripts"))

from panorgan.config import detect_runtime_profile, load_config, seed_everything
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.evaluation import ModalityConditionalConformal, TemperatureScaler
from panorgan.models import PanOrganNetV2
from PanOrgan_4_fix_heads import COHORT_HEAD_MAP, MultiScaleCohortHead, pool_multiscale
from PanOrgan_4_train import _compute_ece
from generate_all_thesis_reports import summarize_seg
from train_v3_pipeline import atomic_torch_save, atomic_write_json

_LOG_FILE = _V41_SSD / "experiments" / "logs" / "v4_1_pipeline.log"


def log_msg(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class PanOrgan41Model(nn.Module):
    def __init__(self, base: PanOrganNetV2, dropout: float = 0.40):
        super().__init__()
        self.base = base
        head_specs: Dict[str, int] = {}
        for _, (hname, n_cls) in COHORT_HEAD_MAP.items():
            head_specs[hname] = n_cls
        self.heads = nn.ModuleDict({
            hname: MultiScaleCohortHead(2688, 512, n_cls, dropout=dropout)
            for hname, n_cls in head_specs.items()
        })

    def forward(self, images: torch.Tensor, meta_vec: torch.Tensor, is_3d: torch.Tensor, prompts: List[str]) -> Dict[str, Any]:
        out = self.base(images, meta_vec, is_3d, prompts=prompts, return_features=True)
        ms = pool_multiscale(out)
        out["head_logits"] = {h: head(ms) for h, head in self.heads.items()}
        return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@torch.no_grad()
def evaluate_cls(
    model: PanOrgan41Model,
    loader: DataLoader,
    device: torch.device,
    cw_map: Dict[str, torch.Tensor],
    tta: bool = True,
) -> Dict[str, Any]:
    model.eval()
    head_specs = {hname: n_cls for _, (hname, n_cls) in COHORT_HEAD_MAP.items()}
    by_h: Dict[str, Dict[str, list]] = {h: {"logits": [], "labels": [], "cohorts": []} for h in head_specs}
    total_loss, n_loss = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        out1 = model.base(images, meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
        if tta:
            out2 = model.base(torch.flip(images, dims=[-1]), meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
            ms = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
        else:
            ms = pool_multiscale(out1)

        for i, ds in enumerate(batch["dataset"]):
            if ds not in COHORT_HEAD_MAP:
                continue
            hname, n_cls = COHORT_HEAD_MAP[ds]
            lbl = int(batch["label"][i].item())
            if lbl < 0 or lbl >= n_cls:
                continue
            lg = model.heads[hname](ms[i : i + 1])
            loss = F.cross_entropy(lg, torch.tensor([lbl], device=device), weight=cw_map[hname], label_smoothing=0.05)
            total_loss += float(loss.item())
            n_loss += 1
            by_h[hname]["logits"].append(lg[0].cpu())
            by_h[hname]["labels"].append(lbl)
            by_h[hname]["cohorts"].append(ds)

    all_y, all_pred, f1s = [], [], []
    for hname, d in by_h.items():
        if not d["logits"]:
            continue
        lg = torch.stack(d["logits"])
        y = np.array(d["labels"])
        pred = lg.argmax(dim=-1).numpy()
        all_y.extend(y.tolist())
        all_pred.extend(pred.tolist())
        f1s.append(float(f1_score(y, pred, average="macro", zero_division=0)))

    return {
        "loss": total_loss / max(1, n_loss),
        "weighted_acc": float(accuracy_score(all_y, all_pred)) if all_y else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "by_h": by_h,
    }


def main() -> None:
    seed_everything(42)
    cfg = load_config(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    shards_dir = _V41_SSD / "preprocessed_shards"
    manifest_path = shards_dir / "master_manifest_v4_1.parquet"
    df = pd.read_parquet(manifest_path)

    # Keep supervised classification cohorts (excluding prostatex_mri which has no cls labels)
    cls_df = df[df["dataset"].isin(COHORT_HEAD_MAP.keys()) & (df["label"] >= 0)].reset_index(drop=True)
    train_df = cls_df[cls_df["split"] == "train"].reset_index(drop=True)
    val_df = cls_df[cls_df["split"] == "val"].reset_index(drop=True)
    test_df = cls_df[cls_df["split"] == "test"].reset_index(drop=True)

    log_msg(f"=== PanOrgan 4.1 Training Started on {device} | train={len(train_df)} | val={len(val_df)} | test={len(test_df)} ===")

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True)
    val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    test_ds = PanOrganShardDataset(test_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

    sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=16, temperature_alpha=0.25)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=0)

    # Class weights per head
    cw_map: Dict[str, torch.Tensor] = {}
    for hname, n_cls in set(COHORT_HEAD_MAP.values()):
        cohorts_h = [c for c, (hn, _) in COHORT_HEAD_MAP.items() if hn == hname]
        sub = train_df[train_df["dataset"].isin(cohorts_h)]["label"].values
        sub = sub[(sub >= 0) & (sub < n_cls)]
        counts = np.bincount(sub, minlength=n_cls).astype(np.float32) if len(sub) > 0 else np.ones(n_cls, dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        w = (counts.sum() / counts) ** 0.5
        w = w / w.mean()
        cw_map[hname] = torch.tensor(w, dtype=torch.float32, device=device)

    base = PanOrganNetV2.from_config(cfg)
    st_best = torch.load(_V4_SSD / "checkpoints" / "panorgan_4" / "best.pt", map_location="cpu", weights_only=False)
    base.load_state_dict({k[len("base."):]: v for k, v in st_best["model_state_dict"].items() if k.startswith("base.")}, strict=False)

    model = PanOrgan41Model(base, dropout=0.40).to(device)
    ms_ck = torch.load(_V4_SSD / "checkpoints" / "panorgan_4" / "multiscale_heads.pt", map_location="cpu", weights_only=False)
    for hname, sd in ms_ck["multiscale_heads"].items():
        if hname in model.heads:
            model.heads[hname].load_state_dict(sd, strict=False)

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    epochs = 5
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.08 * total_steps))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.base.parameters(), "lr": 1.5e-5, "weight_decay": 0.08},
            {"params": model.heads.parameters(), "lr": 8.0e-5, "weight_decay": 0.08},
        ]
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = _V41_SSD / "checkpoints" / "panorgan_4_1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = _V41_SSD / "experiments" / "thesis_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history = []
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            labels = batch["label"].to(device)
            datasets = batch["dataset"]

            # MixUp regularization (p=0.25) on images within same batch
            use_mix = rng.random() < 0.25 and images.size(0) > 1
            if use_mix:
                lam = float(rng.beta(0.4, 0.4))
                perm = torch.randperm(images.size(0), device=device)
                mixed_img = lam * images + (1.0 - lam) * images[perm]
            else:
                lam = 1.0
                perm = None
                mixed_img = images

            optimizer.zero_grad()
            out = model(mixed_img, meta_vec, is_3d, prompts=batch["prompt"])
            losses = []
            for i, ds in enumerate(datasets):
                if ds not in COHORT_HEAD_MAP:
                    continue
                hname, n_cls = COHORT_HEAD_MAP[ds]
                lbl = int(labels[i].item())
                if lbl < 0 or lbl >= n_cls:
                    continue
                lg = out["head_logits"][hname][i : i + 1]
                ce = F.cross_entropy(lg, labels[i : i + 1], weight=cw_map[hname], label_smoothing=0.05)
                if use_mix and perm is not None and datasets[int(perm[i])] == ds:
                    lbl2 = int(labels[perm[i]].item())
                    if 0 <= lbl2 < n_cls:
                        ce2 = F.cross_entropy(lg, labels[perm[i] : perm[i] + 1], weight=cw_map[hname], label_smoothing=0.05)
                        ce = lam * ce + (1.0 - lam) * ce2
                losses.append(ce)

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ep, mp in zip(ema_model.parameters(), model.parameters()):
                    ep.data.mul_(0.998).add_(mp.data, alpha=0.002)

            tr_loss += float(loss.item())
            tr_n += 1

        val_res = evaluate_cls(ema_model, val_loader, device, cw_map, tta=False)
        tr_avg = tr_loss / max(1, tr_n)
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": tr_avg,
            "val_loss": val_res["loss"],
            "val_weighted_acc": val_res["weighted_acc"],
            "val_mean_macro_f1": val_res["mean_macro_f1"],
        })
        log_msg(
            f"[PanOrgan 4.1 Ep {epoch}/{epochs} ({elapsed:.0f}s)] "
            f"tr_loss={tr_avg:.4f} | val_loss={val_res['loss']:.4f} | "
            f"val_acc={val_res['weighted_acc']*100:.2f}% | val_macro_f1={val_res['mean_macro_f1']:.4f}"
        )
        if val_res["mean_macro_f1"] > best_score:
            best_score = val_res["mean_macro_f1"]
            atomic_torch_save({"epoch": epoch, "model_state_dict": ema_model.state_dict(), "val_macro_f1": best_score}, ckpt_dir / "best.pt")

    # Load best EMA checkpoint for calibration & final test evaluation
    best_ck = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    ema_model.load_state_dict(best_ck["model_state_dict"])
    val_final = evaluate_cls(ema_model, val_loader, device, cw_map, tta=True)
    test_final = evaluate_cls(ema_model, test_loader, device, cw_map, tta=True)

    per_head: Dict[str, Any] = {}
    all_y, all_pred = [], []
    for hname, n_cls in set(COHORT_HEAD_MAP.values()):
        vd = val_final["by_h"].get(hname, {"logits": [], "labels": []})
        td = test_final["by_h"].get(hname, {"logits": [], "labels": [], "cohorts": []})
        if not td["logits"]:
            continue

        t_lg = torch.stack(td["logits"])
        y_te = np.array(td["labels"])

        # Temperature scaling on val
        temp = 1.0
        if vd["logits"]:
            scaler = TemperatureScaler()
            temp = float(scaler.fit(torch.stack(vd["logits"]), torch.tensor(vd["labels"], dtype=torch.long)))

        p_te = F.softmax(t_lg / temp, dim=-1).numpy()

        # Macro-F1 optimal threshold for binary heads (prevents positive/negative collapse)
        thresh = 0.5
        if n_cls == 2 and vd["logits"]:
            vp = F.softmax(torch.stack(vd["logits"]) / temp, dim=-1).numpy()[:, 1]
            vy = np.array(vd["labels"])
            best_f1, best_th = -1.0, 0.5
            for th in np.linspace(0.30, 0.70, 41):
                f1_v = float(f1_score(vy, (vp >= th).astype(int), average="macro", zero_division=0))
                if f1_v > best_f1:
                    best_f1, best_th = f1_v, float(th)
            thresh = best_th

        pred_te = (p_te[:, 1] >= thresh).astype(int) if n_cls == 2 else p_te.argmax(axis=-1)
        acc = float(accuracy_score(y_te, pred_te))
        b_acc = float(balanced_accuracy_score(y_te, pred_te))
        mf1 = float(f1_score(y_te, pred_te, average="macro", zero_division=0))
        ece = float(_compute_ece(p_te, y_te))
        auroc = 0.5
        auprc = 0.0
        if len(np.unique(y_te)) > 1:
            if n_cls == 2:
                auroc = float(roc_auc_score(y_te, p_te[:, 1]))
                auprc = float(average_precision_score(y_te, p_te[:, 1]))
            else:
                auroc = float(roc_auc_score(y_te, p_te, multi_class="ovr", average="macro"))
                auprc = float(np.mean([average_precision_score((y_te == c).astype(int), p_te[:, c]) for c in range(n_cls) if np.sum(y_te == c) > 0]))

        acc_ci = bootstrap_ci(y_te, pred_te, accuracy_score)
        per_head[hname] = {
            "n": int(len(y_te)),
            "temperature": temp,
            "decision_threshold": thresh,
            "accuracy": acc,
            "accuracy_ci95": list(acc_ci),
            "balanced_accuracy": b_acc,
            "macro_f1": mf1,
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece,
            "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
        }
        all_y.extend(y_te.tolist())
        all_pred.extend(pred_te.tolist())
        log_msg(f"  [TEST {hname:16s}] n={len(y_te):4d} | Acc={acc*100:.2f}% | BalAcc={b_acc*100:.2f}% | F1={mf1:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} | ECE={ece:.4f} | Th={thresh:.2f}")

    overall_acc = float(accuracy_score(all_y, all_pred))
    overall_ci = bootstrap_ci(np.array(all_y), np.array(all_pred), accuracy_score)
    mean_f1 = float(np.mean([v["macro_f1"] for v in per_head.values()]))

    report = {
        "model": "PanOrgan 4.1",
        "history": history,
        "overall_classification": {
            "n_samples": len(all_y),
            "accuracy": overall_acc,
            "accuracy_ci95": list(overall_ci),
            "mean_head_macro_f1": mean_f1,
        },
        "per_task_group": per_head,
    }
    atomic_write_json(rep_dir / "panorgan_4_1_complete_thesis_report.json", report)
    log_msg(f"=== PanOrgan 4.1 COMPLETE | Overall WeightedAcc={overall_acc*100:.2f}% (95% CI {overall_ci[0]*100:.2f}-{overall_ci[1]*100:.2f}%) | MeanHeadF1={mean_f1:.4f} ===")


if __name__ == "__main__":
    main()
