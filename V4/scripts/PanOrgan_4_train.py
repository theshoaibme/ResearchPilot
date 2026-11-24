#!/usr/bin/env python3
"""PanOrgan 4: Task-Specific Multi-Head Architecture + Per-Group Calibration + FP16 Export."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_4"))
os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from train_v3_pipeline import atomic_torch_save, atomic_write_json, log_msg


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    acc = (preds == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += float(m.mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)

TASK_GROUP_MAP: Dict[str, str] = {
    "pulmonology_cxr": "cxr_4class",
    "pulmonology_cxr_ext": "cxr_4class",
    "nih_cxr14": "cxr_4class",
    "busi": "busi_3class",
    "cmmd_mg": "mg_binary",
    "cbis_ddsm": "mg_binary",
    "kidney_ct_abn": "ct_binary",
    "tcga_kirc_ct": "ct_binary",
    "tcga_lgg": "mri_us_binary",
    "prostatex_mri": "mri_us_binary",
    "ultrasound_nerve": "mri_us_binary",
}

TASK_NUM_CLASSES: Dict[str, int] = {
    "cxr_4class": 4,
    "busi_3class": 3,
    "mg_binary": 2,
    "ct_binary": 2,
    "mri_us_binary": 2,
}


class PanOrgan4MultiHead(nn.Module):
    """Wraps pretrained PanOrganNetV2 backbone with dedicated task-group classification heads."""

    def __init__(self, base: PanOrganNetV2, embed_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.base = base
        self.task_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, n_cls),
            )
            for name, n_cls in TASK_NUM_CLASSES.items()
        })

    def forward(
        self,
        images: torch.Tensor,
        meta_vec: torch.Tensor,
        is_3d: torch.Tensor,
        prompts: List[str],
    ) -> Dict[str, Any]:
        out = self.base(images, meta_vec, is_3d, prompts=prompts)
        emb = out["embedding"]
        out["task_logits"] = {name: head(emb) for name, head in self.task_heads.items()}
        return out


@torch.no_grad()
def evaluate_v4(
    model: PanOrgan4MultiHead,
    loader: DataLoader,
    device: torch.device,
    seg_crit: nn.Module,
) -> Dict[str, Any]:
    model.eval()
    val_loss = 0.0
    n_batches = 0
    all_preds, all_labels, all_groups = [], [], []
    dice_scores: List[float] = []

    for batch in loader:
        images = batch["image"].to(device)
        meta_vec = batch["meta_vec"].to(device)
        is_3d = batch["is_3d"].to(device)
        has_mask = batch["has_mask"].to(device)
        labels = batch["label"].to(device)
        datasets = batch["dataset"]

        out = model(images, meta_vec, is_3d, prompts=batch["prompt"])
        loss = torch.zeros((), device=device)

        if has_mask.any():
            seg_l = seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])
            loss = loss + seg_l
            probs_m = torch.sigmoid(out["seg_logits"][has_mask, 0])
            preds_m = (probs_m > 0.5).float()
            gt_m = batch["mask"].to(device)[has_mask]
            inter = (preds_m * gt_m).sum(dim=(1, 2))
            union = preds_m.sum(dim=(1, 2)) + gt_m.sum(dim=(1, 2))
            d = ((2.0 * inter + 1e-6) / (union + 1e-6)).cpu().tolist()
            dice_scores.extend(d)

        # Route each sample to its dedicated task head
        cls_losses = []
        for i, ds in enumerate(datasets):
            lbl = int(labels[i].item())
            if lbl < 0:
                continue
            tg = TASK_GROUP_MAP[ds]
            logits_i = out["task_logits"][tg][i : i + 1]
            n_c = TASK_NUM_CLASSES[tg]
            if lbl >= n_c:
                continue
            cls_losses.append(F.cross_entropy(logits_i, labels[i : i + 1]))
            all_preds.append(int(logits_i.argmax(dim=-1).item()))
            all_labels.append(lbl)
            all_groups.append(ds)

        if cls_losses:
            loss = loss + 3.0 * torch.stack(cls_losses).mean()
        val_loss += float(loss.item())
        n_batches += 1

    val_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    val_acc = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0
    val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0)) if all_labels else 0.0
    comp = 0.5 * val_dice + 0.5 * val_f1
    return {
        "val_loss": val_loss / max(n_batches, 1),
        "val_dice": val_dice,
        "val_macro_f1": val_f1,
        "val_acc": val_acc,
        "composite": comp,
    }


@torch.no_grad()
def run_v4_test_eval(
    model: PanOrgan4MultiHead,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_mgr: ShardCacheManager,
    mod_map: Dict[str, int],
    runtime: Any,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    def _collect(df: pd.DataFrame, mnar_trunc: float = 0.0) -> Dict[str, Any]:
        ds = PanOrganShardDataset(df, cache_mgr, mod_map, target_hw=(256, 256), augment=False, mnar_truncation=mnar_trunc)
        loader = DataLoader(ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)
        by_group: Dict[str, Dict[str, list]] = {tg: {"logits": [], "labels": [], "cohorts": []} for tg in TASK_NUM_CLASSES}
        seg_records: List[Dict[str, Any]] = []
        for batch in loader:
            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            has_mask = batch["has_mask"].to(device)
            labels = batch["label"]
            datasets = batch["dataset"]
            out = model(images, meta_vec, is_3d, prompts=batch["prompt"])

            if has_mask.any():
                probs_m = torch.sigmoid(out["seg_logits"][has_mask, 0]).cpu().numpy()
                gt_m = batch["mask"][has_mask].numpy()
                has_idx = torch.where(has_mask)[0].cpu().tolist()
                for k, orig_i in enumerate(has_idx):
                    p_bin = (probs_m[k] > 0.5).astype(np.float32)
                    g_bin = gt_m[k].astype(np.float32)
                    inter = float((p_bin * g_bin).sum())
                    union = float(p_bin.sum() + g_bin.sum() - inter)
                    dice = (2.0 * inter + 1e-6) / (p_bin.sum() + g_bin.sum() + 1e-6)
                    iou = (inter + 1e-6) / (union + 1e-6)
                    seg_records.append({"dataset": datasets[orig_i], "dice": dice, "iou": iou, "hd95_mm": 0.0})

            for i, d_name in enumerate(datasets):
                lbl = int(labels[i].item())
                if lbl < 0:
                    continue
                tg = TASK_GROUP_MAP[d_name]
                if lbl >= TASK_NUM_CLASSES[tg]:
                    continue
                by_group[tg]["logits"].append(out["task_logits"][tg][i].cpu())
                by_group[tg]["labels"].append(lbl)
                by_group[tg]["cohorts"].append(d_name)
        return {"by_group": by_group, "seg_records": seg_records}

    val_res = _collect(val_df, 0.0)
    test_res = _collect(test_df, 0.0)
    seg_20 = summarize_seg(_collect(test_df, 0.20)["seg_records"], n_boot=200)
    seg_40 = summarize_seg(_collect(test_df, 0.40)["seg_records"], n_boot=200)

    per_task_report: Dict[str, Any] = {}
    all_preds, all_trues = [], []
    per_cohort_acc: Dict[str, Any] = {}

    for tg in TASK_NUM_CLASSES:
        if not test_res["by_group"][tg]["logits"]:
            continue
        v_logits = torch.stack(val_res["by_group"][tg]["logits"])
        v_labels = np.array(val_res["by_group"][tg]["labels"])
        t_logits = torch.stack(test_res["by_group"][tg]["logits"])
        t_labels = np.array(test_res["by_group"][tg]["labels"])
        t_cohorts = np.array(test_res["by_group"][tg]["cohorts"])

        scaler = TemperatureScaler()
        t_opt = scaler.fit(v_logits, torch.from_numpy(v_labels))
        probs = scaler.transform_probs(t_logits).numpy()
        preds = probs.argmax(axis=-1)

        cp = ModalityConditionalConformal(alpha=0.10)
        cp.fit(scaler.transform_probs(v_logits), v_labels, val_res["by_group"][tg]["cohorts"])
        cp_cov = cp.evaluate_coverage(scaler.transform_probs(t_logits), t_labels, t_cohorts.tolist())

        all_preds.extend(preds.tolist())
        all_trues.extend(t_labels.tolist())

        for c in sorted(set(t_cohorts.tolist())):
            idx = t_cohorts == c
            per_cohort_acc[c] = {"n": int(idx.sum()), "accuracy": round(float((preds[idx] == t_labels[idx]).mean()), 4)}

        auc = float(roc_auc_score(t_labels, probs[:, 1])) if TASK_NUM_CLASSES[tg] == 2 else float(
            roc_auc_score(t_labels, probs, multi_class="ovr", average="macro")
        )
        ece_val = _compute_ece(probs, t_labels)
        per_task_report[tg] = {
            "n": int(len(t_labels)),
            "temperature": float(t_opt),
            "accuracy": float(accuracy_score(t_labels, preds)),
            "macro_f1": float(f1_score(t_labels, preds, average="macro", zero_division=0)),
            "auroc": auc,
            "ece": ece_val,
            "confusion_matrix": confusion_matrix(t_labels, preds).tolist(),
            "per_class_report": classification_report(t_labels, preds, output_dict=True, zero_division=0),
            "mc2_conformal": cp_cov,
        }

    overall_cls = {
        "n_samples": int(len(all_trues)),
        "accuracy": float(accuracy_score(all_trues, all_preds)),
        "macro_f1": float(f1_score(all_trues, all_preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(all_trues, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_trues, all_preds, average="macro", zero_division=0)),
    }
    seg_0 = summarize_seg(test_res["seg_records"], n_boot=200)
    return {
        "model": "PanOrgan_4_MultiHead",
        "overall_classification": overall_cls,
        "per_task_group": per_task_report,
        "per_cohort_accuracy": per_cohort_acc,
        "segmentation": seg_0,
        "mnar_stress_curve": {"0": seg_0, "20": seg_20, "40": seg_40},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PanOrgan 4 Multi-Head Training")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--warm-start", type=str, default=str(_V3_SSD / "checkpoints" / "v32_balanced" / "best.pt"))
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    seed_everything(cfg["project"]["seed"], deterministic=cfg["project"]["deterministic"])
    runtime = detect_runtime_profile(cfg)
    device = torch.device(runtime.device)

    shards_dir = _V3_SSD / "preprocessed_shards"
    ckpt_dir = _SSD_ROOT / "checkpoints" / "panorgan_4"
    exp_dir = _SSD_ROOT / "experiments"
    log_file = exp_dir / "panorgan_4_pipeline.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = shards_dir / "master_manifest_v4.parquet"
    if not manifest_path.exists():
        manifest_path = shards_dir / "master_manifest.parquet"
    manifest_df = pd.read_parquet(manifest_path)

    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"].isin(["test", "external_test"])].reset_index(drop=True)

    sup_train = train_df[(train_df["has_mask"]) | (train_df["label"] >= 0)].reset_index(drop=True)
    sup_val = val_df[(val_df["has_mask"]) | (val_df["label"] >= 0)].reset_index(drop=True)
    log_msg(f"PanOrgan 4 Supervised -> Train: {len(sup_train)} | Val: {len(sup_val)} | Test: {len(test_df)}", log_file)

    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}
    cache_mgr = ShardCacheManager(shards_dir, shards_dir, max_cache_gb=100.0)
    seg_crit = DiceFocalBoundaryLoss()

    train_ds = PanOrganShardDataset(sup_train, cache_mgr, mod_map, target_hw=(256, 256), augment=True, mnar_truncation=0.20)
    val_ds = PanOrganShardDataset(sup_val, cache_mgr, mod_map, target_hw=(256, 256), augment=False)
    sampler = BalancedModalityBatchSampler(sup_train["dataset"].tolist(), batch_size=runtime.batch_size, temperature_alpha=0.30)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=runtime.batch_size, shuffle=False, num_workers=0)

    base_model = PanOrganNetV2.from_config(cfg)
    ws_path = Path(args.warm_start)
    if not ws_path.exists():
        ws_path = _SSD_ROOT / "checkpoints" / "v31_5modal_mixup" / "best.pt"
    st = torch.load(ws_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(st["model_state_dict"], strict=False)
    log_msg(f"Warm-started PanOrgan 4 backbone from {ws_path}", log_file)

    model = PanOrgan4MultiHead(base_model, embed_dim=int(cfg["model"].get("embed_dim", 384))).to(device)
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"

    # Differential LR: faster learning on fresh task heads (2e-4), gentle on pretrained backbone (3e-5)
    optimizer = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": 3.0e-5},
        {"params": model.task_heads.parameters(), "lr": 2.0e-4},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_comp = -1.0
    no_improve = 0
    resume_b = -1
    global_step = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_dice": [], "val_f1": [], "val_acc": [], "composite": []}

    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_comp = float(ck.get("best_composite", -1.0))
        no_improve = int(ck.get("no_improve", 0))
        global_step = int(ck.get("global_step", 0))
        history = ck.get("history", history)
        if ck.get("mid_epoch", False):
            start_epoch = int(ck["epoch"])
            resume_b = int(ck.get("batch_idx", -1))
            log_msg(f"[RESUME V4] Epoch {start_epoch}/{args.epochs} from batch {resume_b + 1}", log_file)
        else:
            start_epoch = int(ck["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n_b = 0.0, 0

        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[PanOrgan 4] Ep {epoch}/{args.epochs}", leave=False)):
            if epoch == start_epoch and b_idx <= resume_b:
                continue
            images = batch["image"].to(device)
            meta_vec = batch["meta_vec"].to(device)
            is_3d = batch["is_3d"].to(device)
            has_mask = batch["has_mask"].to(device)
            labels = batch["label"].to(device)
            multilabel = batch["multilabel"].to(device)
            datasets = batch["dataset"]

            out = model(images, meta_vec, is_3d, prompts=batch["prompt"])
            total = torch.zeros((), device=device)

            if has_mask.any():
                total = total + 1.0 * seg_crit(out["seg_logits"][has_mask], batch["mask"].to(device)[has_mask])

            cls_losses = []
            for i, d_name in enumerate(datasets):
                lbl = int(labels[i].item())
                if lbl < 0:
                    continue
                tg = TASK_GROUP_MAP[d_name]
                if lbl >= TASK_NUM_CLASSES[tg]:
                    continue
                cls_losses.append(F.cross_entropy(out["task_logits"][tg][i : i + 1], labels[i : i + 1], label_smoothing=0.05))
            if cls_losses:
                total = total + 3.0 * torch.stack(cls_losses).mean()

            valid_ml = multilabel >= 0
            if valid_ml.any():
                total = total + 1.0 * F.binary_cross_entropy_with_logits(out["multilabel_logits"][valid_ml], multilabel[valid_ml])

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
        metrics = evaluate_v4(model, val_loader, device, seg_crit)
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

        st_out = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_composite": best_comp, "history": history}
        atomic_torch_save(st_out, latest_path)
        if is_best:
            atomic_torch_save(st_out, best_path)
            # Export lightweight FP16 weights for fast deployment
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            atomic_torch_save({"model_state_dict": fp16_state}, ckpt_dir / "best_fp16.pt")
        atomic_write_json(ckpt_dir / "training_history.json", history)

        log_msg(
            f"[PanOrgan 4 Ep {epoch}/{args.epochs}] train_loss={tr_l:.4f} | val_loss={metrics['val_loss']:.4f} | "
            f"val_dice={metrics['val_dice']:.4f} | val_f1={metrics['val_macro_f1']:.4f} | "
            f"val_acc={metrics['val_acc']:.4f} (best={best_comp:.4f}, no_improve={no_improve}/{args.patience})",
            log_file,
        )
        if no_improve >= args.patience:
            log_msg(f"[EARLY STOPPING V4] Triggered at Epoch {epoch}.", log_file)
            break

    log_msg("=== Running PanOrgan 4 Per-Task Test Evaluation ===", log_file)
    st_b = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(st_b["model_state_dict"], strict=False)
    report = run_v4_test_eval(model, val_df, test_df, cache_mgr, mod_map, runtime, device)
    out_dir = exp_dir / "thesis_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "panorgan_4_complete_thesis_report.json", report)
    log_msg(f"=== PANORGAN 4 COMPLETED: Acc={report['overall_classification']['accuracy']:.4f} | Dice={report['segmentation']['overall']['dice']['mean']:.4f} ===", log_file)


if __name__ == "__main__":
    main()
