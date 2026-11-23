"""Fault-tolerant Colab Pro training engine for Stage-1 SSL and Stage-2 Multi-Task Pan-Organ Net V2.

Key engineering features:
- Automatic mixed precision (bfloat16 on A100/L4, float16 + GradScaler on T4, float32 on CPU).
- Exponential Moving Average (EMA) teacher for Stage-1 MA-UGM uncertainty guidance.
- Gradient accumulation and norm clipping for fixed effective batch sizes across GPU tiers.
- Atomic checkpoint saving (write tmp -> rename) directly to Google Drive every epoch & N steps.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from panorgan.config import RuntimeProfile
from panorgan.models.decoders import AsymmetricFocalLoss, DiceFocalBoundaryLoss


class ColabResilientTrainer:
    """Unified trainer supporting Stage-1 SSL (MA-UGM + SACR) and Stage-2 Multi-Task training."""

    def __init__(
        self,
        model: nn.Module,
        runtime: RuntimeProfile,
        checkpoint_dir: str | Path,
        lr: float = 1e-4,
        backbone_lr_mult: float = 0.2,
        weight_decay: float = 0.02,
        max_grad_norm: float = 1.0,
        ema_decay: float = 0.996,
        moe_aux_weight: float = 0.02,
        seg_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        sacr_loss_weight: float = 0.25,
        checkpoint_every_steps: int = 500,
        patience: int = 7,
    ):
        self.device = torch.device(runtime.device)
        self.runtime = runtime
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        self.moe_aux_weight = moe_aux_weight
        self.seg_loss_weight = seg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.sacr_loss_weight = sacr_loss_weight
        self.checkpoint_every_steps = checkpoint_every_steps
        self.patience = patience

        # Differential learning rates: lower LR on pretrained encoder, full LR on new modules
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone.encoder" in name and "lora_" not in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        self.use_amp = self.device.type == "cuda" and runtime.amp_dtype in ("bfloat16", "float16")
        self.amp_dtype = torch.bfloat16 if runtime.amp_dtype == "bfloat16" else torch.float16
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        # EMA teacher for MA-UGM uncertainty guidance
        self.ema_model: Optional[nn.Module] = None
        self.seg_criterion = DiceFocalBoundaryLoss()
        self.multilabel_criterion = AsymmetricFocalLoss()

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "seg_loss": [],
            "cls_loss": [],
            "sacr_loss": [],
            "epoch_sec": [],
        }
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.global_step = 0
        self._resume_mid_epoch_state: Optional[Dict[str, Any]] = None

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=True)
        return contextlib.nullcontext()

    def init_ema_teacher(self) -> None:
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        d = self.ema_decay
        for p_ema, p_cur in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(d).add_(p_cur.data, alpha=1.0 - d)

    def _save_atomic(self, state: Dict[str, Any], filename: str) -> Path:
        target = self.checkpoint_dir / filename
        tmp = self.checkpoint_dir / f".{filename}.tmp"
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        return target

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        tag: str = "latest.pt",
        mid_epoch: bool = False,
        batch_idx: int = -1,
        running: Optional[Dict[str, float]] = None,
        n_batches: int = 0,
        scheduler: Optional[Any] = None,
    ) -> None:
        state: Dict[str, Any] = {
            "epoch": epoch,
            "mid_epoch": mid_epoch,
            "batch_idx": batch_idx,
            "running": running,
            "n_batches": n_batches,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "history": self.history,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        self._save_atomic(state, tag)
        if is_best:
            self._save_atomic(state, "best.pt")

        log_path = self.checkpoint_dir / "training_history.json"
        log_tmp = self.checkpoint_dir / ".training_history.json.tmp"
        with open(log_tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.history, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(log_tmp, log_path)

    def resume_if_exists(self, filename: str = "latest.pt") -> int:
        ckpt_path = self.checkpoint_dir / filename
        if not ckpt_path.exists():
            self._resume_mid_epoch_state = None
            return 1
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.ema_model is not None and "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        self.history = ckpt.get("history", self.history)

        if ckpt.get("mid_epoch", False) and int(ckpt.get("batch_idx", -1)) >= 0:
            self._resume_mid_epoch_state = {
                "epoch": int(ckpt["epoch"]),
                "batch_idx": int(ckpt["batch_idx"]),
                "running": ckpt.get("running"),
                "n_batches": int(ckpt.get("n_batches", 0)),
                "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
            }
            return int(ckpt["epoch"])

        completed_epochs = len(self.history.get("train_loss", []))
        next_epoch = int(ckpt["epoch"]) if ("mid_epoch" not in ckpt and completed_epochs < int(ckpt["epoch"])) else int(ckpt["epoch"]) + 1
        self._resume_mid_epoch_state = {
            "epoch": next_epoch,
            "batch_idx": -1,
            "running": None,
            "n_batches": 0,
            "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
        }
        return next_epoch

    def _compute_multitask_loss(self, batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        total = torch.zeros((), device=self.device)
        seg_loss = torch.zeros((), device=self.device)
        cls_loss = torch.zeros((), device=self.device)
        sacr_loss = torch.zeros((), device=self.device)

        has_mask = batch["has_mask"].to(self.device)
        if has_mask.any():
            masks = batch["mask"].to(self.device)[has_mask]
            seg_logits = out["seg_logits"][has_mask]
            seg_loss = self.seg_criterion(seg_logits, masks)
            total = total + self.seg_loss_weight * seg_loss

        multilabel = batch["multilabel"].to(self.device)
        has_ml = (multilabel >= 0).any(dim=1)
        if has_ml.any():
            ml_loss = self.multilabel_criterion(out["multilabel_logits"][has_ml], multilabel[has_ml])
            cls_loss = cls_loss + ml_loss

        labels = batch["label"].to(self.device)
        has_mc = (labels >= 0) & (~has_ml)
        if has_mc.any():
            mc_logits = out["multiclass_logits"][has_mc]
            mc_targets = labels[has_mc].clamp(0, mc_logits.shape[1] - 1)
            ce = F.cross_entropy(mc_logits, mc_targets, reduction="none")
            pt = torch.exp(-ce)
            focal_mc = (((1.0 - pt) ** 2.0) * ce).mean()
            cls_loss = cls_loss + focal_mc

        total = total + self.cls_loss_weight * cls_loss

        if out.get("pred_coords") is not None:
            body_coord = batch["body_coord"].to(self.device)
            sacr_loss = F.smooth_l1_loss(out["pred_coords"], body_coord)
            total = total + self.sacr_loss_weight * sacr_loss

        if out.get("moe_aux_loss") is not None:
            total = total + self.moe_aux_weight * out["moe_aux_loss"]

        return {"total": total, "seg": seg_loss, "cls": cls_loss, "sacr": sacr_loss}

    def _step_backward(self, loss: torch.Tensor, step_now: bool) -> None:
        scaled_loss = loss / float(max(self.runtime.grad_accum, 1))
        if self.use_scaler and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            if step_now:
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if step_now:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        mode: str = "multitask",
        warmup_epochs: int = 2,
        resume: bool = True,
    ) -> Dict[str, list]:
        if mode == "ssl" and self.ema_model is None:
            self.init_ema_teacher()

        start_epoch = self.resume_if_exists() if resume else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, num_epochs), eta_min=1e-6
        )
        if resume and self._resume_mid_epoch_state is not None:
            sched_state = self._resume_mid_epoch_state.get("scheduler_state_dict")
            if sched_state is not None:
                scheduler.load_state_dict(sched_state)

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            resume_batch_idx = -1
            running = {"total": 0.0, "seg": 0.0, "cls": 0.0, "sacr": 0.0}
            n_batches = 0
            if (
                resume
                and self._resume_mid_epoch_state is not None
                and self._resume_mid_epoch_state.get("epoch") == epoch
                and int(self._resume_mid_epoch_state.get("batch_idx", -1)) >= 0
            ):
                resume_batch_idx = int(self._resume_mid_epoch_state["batch_idx"])
                if self._resume_mid_epoch_state.get("running"):
                    running = dict(self._resume_mid_epoch_state["running"])
                n_batches = int(self._resume_mid_epoch_state.get("n_batches", 0))
                print(f"[RESUME] Continuing {mode.upper()} Epoch {epoch}/{num_epochs} from batch step {resume_batch_idx + 1} (global_step={self.global_step})")
                self._resume_mid_epoch_state = None

            # Smooth ramp for EMA teacher uncertainty blend in Stage-1 SSL
            ema_blend = min(0.7, max(0.0, (epoch - warmup_epochs) / max(float(num_epochs - warmup_epochs), 1.0)))

            non_blocking = self.device.type == "cuda"
            for b_idx, batch in enumerate(tqdm(train_loader, desc=f"[{mode.upper()}] Ep {epoch}/{num_epochs}", leave=False)):
                if b_idx <= resume_batch_idx:
                    continue

                images = batch["image"].to(self.device, non_blocking=non_blocking)
                meta_vec = batch["meta_vec"].to(self.device, non_blocking=non_blocking)
                is_3d = batch["is_3d"].to(self.device, non_blocking=non_blocking)

                teacher_unc = None
                if mode == "ssl" and self.ema_model is not None and ema_blend > 0.0:
                    with torch.no_grad():
                        t_out = self.ema_model.forward_ssl(
                            images, meta_vec, is_3d, saliency=batch["saliency"].to(self.device)
                        )
                        teacher_unc = t_out["recon_err_map"]

                with self._autocast_ctx():
                    if mode == "ssl":
                        out = self.model.forward_ssl(
                            images,
                            meta_vec,
                            is_3d,
                            saliency=batch["saliency"].to(self.device, non_blocking=non_blocking),
                            teacher_uncertainty=teacher_unc,
                            ema_blend=ema_blend,
                        )
                        sacr_l = (
                            F.smooth_l1_loss(out["pred_coords"], batch["body_coord"].to(self.device))
                            if out.get("pred_coords") is not None
                            else torch.zeros((), device=self.device)
                        )
                        loss_dict = {
                            "total": out["recon_loss"]
                            + self.sacr_loss_weight * sacr_l
                            + self.moe_aux_weight * out["moe_aux_loss"],
                            "seg": out["recon_loss"],
                            "cls": torch.zeros((), device=self.device),
                            "sacr": sacr_l,
                        }
                    else:
                        out = self.model(images, meta_vec, is_3d, prompts=batch["prompt"])
                        loss_dict = self._compute_multitask_loss(batch, out)

                step_now = ((b_idx + 1) % self.runtime.grad_accum == 0) or ((b_idx + 1) == len(train_loader))
                self._step_backward(loss_dict["total"], step_now=step_now)

                for k in running:
                    running[k] += float(loss_dict[k].detach().item())
                n_batches += 1

                if step_now:
                    self.global_step += 1
                    if mode == "ssl":
                        self._update_ema()
                    if self.checkpoint_every_steps > 0 and (self.global_step % self.checkpoint_every_steps == 0):
                        self.save_checkpoint(
                            epoch,
                            is_best=False,
                            mid_epoch=True,
                            batch_idx=b_idx,
                            running=running,
                            n_batches=n_batches,
                            scheduler=scheduler,
                        )

            scheduler.step()
            train_loss = running["total"] / max(n_batches, 1)
            val_loss = self.evaluate_loss(val_loader, mode=mode) if val_loader is not None else train_loss

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["seg_loss"].append(running["seg"] / max(n_batches, 1))
            self.history["cls_loss"].append(running["cls"] / max(n_batches, 1))
            self.history["sacr_loss"].append(running["sacr"] / max(n_batches, 1))
            self.history["epoch_sec"].append(round(time.time() - t0, 2))

            self.save_checkpoint(epoch, is_best=is_best, mid_epoch=False, batch_idx=-1, scheduler=scheduler)

            if self.epochs_no_improve >= self.patience:
                break

        return self.history

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader, mode: str = "multitask") -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            meta_vec = batch["meta_vec"].to(self.device)
            is_3d = batch["is_3d"].to(self.device)
            with self._autocast_ctx():
                if mode == "ssl":
                    out = self.model.forward_ssl(images, meta_vec, is_3d, saliency=batch["saliency"].to(self.device))
                    loss = out["recon_loss"]
                else:
                    out = self.model(images, meta_vec, is_3d, prompts=batch["prompt"])
                    loss = self._compute_multitask_loss(batch, out)["total"]
            total += float(loss.item())
            count += 1
        return total / max(count, 1)
