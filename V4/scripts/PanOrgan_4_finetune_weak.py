#!/usr/bin/env python3
"""End-to-End Fine-Tuning on Weak Cohorts (cmmd_mg, cbis_ddsm, kidney_ct_abn, tcga_kirc_ct, ultrasound_nerve)
with unfrozen PanOrganNetV2 backbone + MultiScaleCohortHead (keeps strong CXR/LGG/BUSI/Seg intact).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from torch.utils.data import DataLoader
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
from panorgan.data import BalancedModalityBatchSampler, PanOrganShardDataset, ShardCacheManager
from panorgan.evaluation import ModalityConditionalConformal, TemperatureScaler
from panorgan.models import PanOrganNetV2
from PanOrgan_4_fix_heads import COHORT_HEAD_MAP, MultiScaleCohortHead, pool_multiscale
from PanOrgan_4_train import _compute_ece, log_msg
from train_v3_pipeline import atomic_torch_save, atomic_write_json

WEAK_COHORTS = ["cmmd_mg", "cbis_ddsm", "kidney_ct_abn", "tcga_kirc_ct", "ultrasound_nerve"]


class WeakCohortEndToEndNet(nn.Module):
    def __init__(self, base: PanOrganNetV2, heads: Dict[str, MultiScaleCohortHead]):
        super().__init__()
        self.base = base
        self.heads = nn.ModuleDict(heads)

    def forward(self, images: torch.Tensor, meta_vec: torch.Tensor, is_3d: torch.Tensor, prompts: List[str]) -> Dict[str, Any]:
        out = self.base(images, meta_vec, is_3d, prompts=prompts, return_features=True)
        ms_feat = pool_multiscale(out)
        out["head_logits"] = {h: head(ms_feat) for h, head in self.heads.items()}
        return out


@torch.no_grad()
def eval_weak(model: WeakCohortEndToEndNet, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    by_h: Dict[str, Dict[str, list]] = {h: {"logits": [], "labels": [], "cohorts": []} for h in WEAK_COHORTS}
    for batch in loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        out1 = model.base(images, meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
        out2 = model.base(torch.flip(images, dims=[-1]), meta_vec, is_3d, prompts=batch["prompt"], return_features=True)
        ms_feat = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
        for i, ds in enumerate(batch["dataset"]):
            lbl = int(batch["label"][i].item())
            if lbl < 0 or ds not in model.heads or lbl >= 2:
                continue
            lg = model.heads[ds](ms_feat[i : i + 1])[0].cpu()
            by_h[ds]["logits"].append(lg)
            by_h[ds]["labels"].append(lbl)
            by_h[ds]["cohorts"].append(ds)

    accs, f1s, aucs = [], [], []
    for ds in WEAK_COHORTS:
        if not by_h[ds]["logits"]:
            continue
        lg = torch.stack(by_h[ds]["logits"])
        y = np.array(by_h[ds]["labels"])
        p = F.softmax(lg, dim=-1).numpy()
        pred = p.argmax(axis=-1)
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro", zero_division=0))
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p[:, 1]))
    return {
        "mean_acc": float(np.mean(accs)) if accs else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_auc": float(np.mean(aucs)) if aucs else 0.5,
        "by_h": by_h,
    }


def main():
    cfg = load_config(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml")
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = detect_runtime_profile(cfg)

    shards_dir = _V3_SSD / "preprocessed_shards"
    ckpt_dir = _SSD_ROOT / "checkpoints" / "panorgan_4"
    exp_dir = _SSD_ROOT / "experiments"
    log_file = exp_dir / "panorgan_4_pipeline.log"

    df = pd.read_parquet(shards_dir / "master_manifest_v4.parquet")
    df = df[df["dataset"].isin(WEAK_COHORTS) & (df["label"] >= 0) & (df["label"] < 2)].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    log_msg(f"=== PanOrgan 4 End-to-End Weak-Cohort Fine-Tuning | Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} ===", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    train_ds = PanOrganShardDataset(train_df, cache_mgr, mod_map, target_hw=(256, 256), augment=True)
    val_ds = PanOrganShardDataset(val_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    test_ds = PanOrganShardDataset(test_df, cache_mgr, mod_map, target_hw=(256, 256), augment=False)

    sampler = BalancedModalityBatchSampler(train_df["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.25)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

    base_model = PanOrganNetV2.from_config(cfg)
    st = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    base_sd = {k[len("base."):]: v for k, v in st["model_state_dict"].items() if k.startswith("base.")}
    base_model.load_state_dict(base_sd, strict=False)

    ms_ck = torch.load(ckpt_dir / "multiscale_heads.pt", map_location="cpu", weights_only=False)
    heads: Dict[str, MultiScaleCohortHead] = {}
    for ds in WEAK_COHORTS:
        h = MultiScaleCohortHead(2688, 512, 2, dropout=0.30)
        if ds in ms_ck.get("multiscale_heads", {}):
            h.load_state_dict(ms_ck["multiscale_heads"][ds])
        heads[ds] = h

    model = WeakCohortEndToEndNet(base_model, heads).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.995

    # Per-cohort inverse-sqrt class weights
    cls_weights: Dict[str, torch.Tensor] = {}
    for ds in WEAK_COHORTS:
        sub = train_df[train_df["dataset"] == ds]["label"].values
        c = np.maximum(np.bincount(sub, minlength=2).astype(np.float64), 1.0)
        w = (c.sum() / c) ** 0.5
        cls_weights[ds] = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    epochs = 4
    optimizer = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": 4.5e-5, "weight_decay": 0.04},
        {"params": model.heads.parameters(), "lr": 2.0e-4, "weight_decay": 0.05},
    ])
    total_steps = max(1, (len(train_loader) // max(1, runtime.grad_accum)) * epochs)
    warmup = max(10, int(0.10 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / float(warmup) if s < warmup else max(0.05, 0.5 * (1.0 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup)))),
    )

    best_score = -1.0
    best_path = ckpt_dir / "weak_cohort_specialist.pt"

    for ep in range(1, epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(ep)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n_b = 0.0, 0

        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[PanOrgan 4 Weak-FT] Ep {ep}/{epochs}", leave=False)):
            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            labels = batch["label"].to(device)

            out = model(images, meta_vec, is_3d, prompts=batch["prompt"])
            losses = []
            for i, ds in enumerate(batch["dataset"]):
                lbl = int(labels[i].item())
                if lbl < 0 or ds not in model.heads or lbl >= 2:
                    continue
                lg = out["head_logits"][ds][i : i + 1]
                ce = F.cross_entropy(lg, labels[i : i + 1], weight=cls_weights[ds], label_smoothing=0.03, reduction="none")
                pt = torch.exp(-F.cross_entropy(lg, labels[i : i + 1], reduction="none")).clamp(1e-6, 1.0)
                losses.append(((1.0 - pt) ** 1.5) * ce)
            if not losses:
                continue
            total = torch.stack(losses).mean() + float(cfg["model"]["moe_aux_weight"]) * out["moe_aux_loss"]
            (total / max(1, runtime.grad_accum)).backward()
            run_loss += float(total.detach().item())
            n_b += 1

            if ((b_idx + 1) % runtime.grad_accum == 0) or ((b_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for pe, p in zip(ema_model.parameters(), model.parameters()):
                        pe.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        val_res = eval_weak(ema_model, val_loader, device)
        score = 0.5 * val_res["mean_auc"] + 0.5 * val_res["mean_f1"]
        is_best = score > best_score
        if is_best:
            best_score = score
            atomic_torch_save({"epoch": ep, "model_state_dict": ema_model.state_dict(), "score": best_score}, best_path)
        log_msg(
            f"[PanOrgan 4 Weak-FT Ep {ep}/{epochs}] loss={run_loss/max(n_b,1):.4f} | val_acc={val_res['mean_acc']:.4f} | val_f1={val_res['mean_f1']:.4f} | val_auc={val_res['mean_auc']:.4f} (best={best_score:.4f})",
            log_file,
        )

    # Final calibrated evaluation & merge into panorgan_4_complete_thesis_report.json
    ema_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"], strict=False)
    v_res = eval_weak(ema_model, val_loader, device)["by_h"]
    t_res = eval_weak(ema_model, test_loader, device)["by_h"]

    rep_path = exp_dir / "thesis_reports" / "panorgan_4_complete_thesis_report.json"
    rep = json.loads(rep_path.read_text())

    for ds in WEAK_COHORTS:
        if not t_res[ds]["logits"] or not v_res[ds]["logits"]:
            continue
        vl = torch.stack(v_res[ds]["logits"])
        vy = np.array(v_res[ds]["labels"])
        tl = torch.stack(t_res[ds]["logits"])
        ty = np.array(t_res[ds]["labels"])
        scaler = TemperatureScaler()
        t_opt = scaler.fit(vl, torch.from_numpy(vy))
        vp = np.asarray(scaler.transform_probs(vl))
        tp = np.asarray(scaler.transform_probs(tl))

        best_thr, best_f1 = 0.50, -1.0
        for thr in np.linspace(0.25, 0.75, 51):
            f1_v = float(f1_score(vy, (vp[:, 1] >= thr).astype(int), average="macro", zero_division=0))
            if f1_v > best_f1:
                best_f1, best_thr = f1_v, float(thr)
        preds = (tp[:, 1] >= best_thr).astype(int)

        cp = ModalityConditionalConformal(alpha=0.10)
        cp.fit(vp, vy, v_res[ds]["cohorts"])
        cp_cov = cp.evaluate_coverage(tp, ty, t_res[ds]["cohorts"])
        auc = float(roc_auc_score(ty, tp[:, 1])) if len(np.unique(ty)) > 1 else 0.5
        acc = float(accuracy_score(ty, preds))
        f1 = float(f1_score(ty, preds, average="macro", zero_division=0))
        ece = _compute_ece(tp, ty)

        rep["per_task_group"][ds] = {
            "n": int(len(ty)),
            "temperature": float(t_opt),
            "decision_threshold": float(best_thr),
            "accuracy": acc,
            "macro_f1": f1,
            "auroc": auc,
            "ece": ece,
            "confusion_matrix": confusion_matrix(ty, preds).tolist(),
            "per_class_report": classification_report(ty, preds, output_dict=True, zero_division=0),
            "mc2_conformal": cp_cov,
        }
        rep["per_cohort_accuracy"][ds] = {"n": int(len(ty)), "accuracy": round(acc, 4)}
        log_msg(f"[PanOrgan 4 Weak-FT Final {ds:16s}] n={len(ty):4d} | Acc={acc:.4f} | F1={f1:.4f} | AUROC={auc:.4f} | ECE={ece:.4f}", log_file)

    # Recompute overall classification accuracy weighted across all cohorts
    total_n = sum(v["n"] for v in rep["per_task_group"].values())
    weighted_acc = sum(v["n"] * v["accuracy"] for v in rep["per_task_group"].values()) / max(total_n, 1)
    mean_f1 = float(np.mean([v["macro_f1"] for v in rep["per_task_group"].values()]))
    rep["overall_classification"]["accuracy"] = float(weighted_acc)
    rep["overall_classification"]["macro_f1"] = float(mean_f1)
    atomic_write_json(rep_path, rep)
    log_msg(f"=== PANORGAN 4 WEAK-COHORT FINE-TUNING COMPLETED: WeightedAcc={weighted_acc:.4f} | MeanHeadF1={mean_f1:.4f} ===", log_file)


if __name__ == "__main__":
    main()
