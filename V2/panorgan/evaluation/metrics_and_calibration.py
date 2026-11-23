"""Q1-grade clinical evaluation, calibration, Modality-Conditional Conformal Prediction (MC2),
statistical significance tests, XAI faithfulness metrics, and LaTeX table generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_dice_and_hd95(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    spacing_mm: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, float]:
    """Compute Dice Similarity Coefficient (DSC) and 95th-percentile Hausdorff Distance (HD95 in mm)
    for 2D (H, W) or 3D (D, H, W) binary anatomical masks."""
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    p_sum = int(p.sum())
    t_sum = int(t.sum())

    if p_sum == 0 and t_sum == 0:
        return {"dice": 1.0, "hd95_mm": 0.0}
    if p_sum == 0 or t_sum == 0:
        return {"dice": 0.0, "hd95_mm": float("nan")}

    inter = int(np.logical_and(p, t).sum())
    dice = (2.0 * inter) / float(p_sum + t_sum)

    # Extract surface voxels via binary erosion
    p_surface = np.logical_xor(p, ndimage.binary_erosion(p))
    t_surface = np.logical_xor(t, ndimage.binary_erosion(t))
    if not np.any(p_surface):
        p_surface = p
    if not np.any(t_surface):
        t_surface = t

    sampling = spacing_mm[: p.ndim]
    dt_t = ndimage.distance_transform_edt(~t_surface, sampling=sampling)
    dt_p = ndimage.distance_transform_edt(~p_surface, sampling=sampling)

    dists_p_to_t = dt_t[p_surface]
    dists_t_to_p = dt_p[t_surface]
    all_dists = np.concatenate([dists_p_to_t, dists_t_to_p])
    hd95 = float(np.percentile(all_dists, 95))
    return {"dice": float(dice), "hd95_mm": hd95}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    specificity_targets: Sequence[float] = (0.90, 0.95),
) -> Dict[str, float]:
    """Compute clinical classification metrics emphasizing FN sensitivity alongside AUROC/AUPRC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        pos_prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        y_bin = (y_true > 0).astype(int)
        preds = (pos_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, preds, labels=[0, 1]).ravel()

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        auroc = float(roc_auc_score(y_bin, pos_prob)) if len(np.unique(y_bin)) > 1 else float("nan")
        auprc = float(average_precision_score(y_bin, pos_prob)) if len(np.unique(y_bin)) > 1 else float("nan")

        out = {
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "macro_f1": float(f1_score(y_bin, preds, average="macro", zero_division=0)),
            "accuracy": float((tp + tn) / max(len(y_bin), 1)),
        }
        if len(np.unique(y_bin)) > 1:
            fpr, tpr, _ = roc_curve(y_bin, pos_prob)
            spec_curve = 1.0 - fpr
            for sp_t in specificity_targets:
                valid = np.where(spec_curve >= sp_t)[0]
                out[f"sens_at_{int(sp_t * 100)}spec"] = float(np.max(tpr[valid])) if len(valid) > 0 else 0.0
        return out

    # Multi-class evaluation
    preds = np.argmax(y_prob, axis=1)
    auroc = float("nan")
    try:
        auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return {
        "auroc": auroc,
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "accuracy": float(np.mean(preds == y_true)),
    }


def compute_ece_and_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and multiclass/binary Brier Score."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences >= lo) & (confidences <= hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin])))

    one_hot = np.eye(probs.shape[1])[np.clip(y_true, 0, probs.shape[1] - 1)]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier": brier}


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling optimizing NLL on the held-out calibration split."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        logits = logits.detach().float()
        labels = labels.detach().long()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            t = self.temperature.clamp(min=0.05, max=10.0)
            loss = F.cross_entropy(logits / t, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)
        return float(self.temperature.item())

    def transform_probs(self, logits: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            t = self.temperature.clamp(min=0.05, max=10.0)
            return F.softmax(logits.float() / t, dim=-1).cpu().numpy()


class ModalityConditionalConformal:
    """Modality-Conditional Split Conformal Prediction (MC2-CP).

    Stratifies calibration quantiles by (modality, missingness_pattern) via a Mondrian
    taxonomy so finite-sample coverage >= (1 - alpha) holds even when scans or modalities
    are truncated/missing.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.group_quantiles: Dict[str, float] = {}
        self.global_quantile: float = 1.0

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.0
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / float(n))
        return float(np.quantile(scores, q_level, method="higher"))

    def fit(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        cal_groups: Sequence[str],
    ) -> Dict[str, float]:
        probs = np.asarray(cal_probs, dtype=np.float64)
        labels = np.asarray(cal_labels, dtype=int)
        groups = np.asarray(cal_groups)
        scores = 1.0 - probs[np.arange(len(labels)), labels]
        self.global_quantile = self._conformal_quantile(scores, self.alpha)

        self.group_quantiles = {}
        for g in np.unique(groups):
            g_scores = scores[groups == g]
            self.group_quantiles[str(g)] = self._conformal_quantile(g_scores, self.alpha)
        return dict(self.group_quantiles)

    def predict_sets(
        self,
        test_probs: np.ndarray,
        test_groups: Sequence[str],
    ) -> List[List[int]]:
        probs = np.asarray(test_probs, dtype=np.float64)
        pred_sets: List[List[int]] = []
        for p_row, g in zip(probs, test_groups):
            q = self.group_quantiles.get(str(g), self.global_quantile)
            included = np.where((1.0 - p_row) <= q)[0].tolist()
            if not included:
                included = [int(np.argmax(p_row))]
            pred_sets.append(included)
        return pred_sets

    def evaluate_coverage(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
        test_groups: Sequence[str],
    ) -> Dict[str, object]:
        pred_sets = self.predict_sets(test_probs, test_groups)
        labels = np.asarray(test_labels, dtype=int)
        groups = np.asarray(test_groups)

        covered = np.array([int(y in pset) for y, pset in zip(labels, pred_sets)], dtype=np.float64)
        set_sizes = np.array([len(pset) for pset in pred_sets], dtype=np.float64)
        abstention = np.array([int(len(pset) > 1) for pset in pred_sets], dtype=np.float64)

        per_group = {}
        for g in np.unique(groups):
            m = groups == g
            per_group[str(g)] = {
                "coverage": float(np.mean(covered[m])),
                "avg_set_size": float(np.mean(set_sizes[m])),
                "abstention_rate": float(np.mean(abstention[m])),
                "n": int(np.sum(m)),
            }

        return {
            "target_coverage": 1.0 - self.alpha,
            "marginal_coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(set_sizes)),
            "abstention_rate": float(np.mean(abstention)),
            "per_group": per_group,
        }


def compute_mnar_degradation(full_score: float, truncated_score: float) -> float:
    """Relative degradation rate Delta_MNAR (%) under simulated FOV truncation."""
    if abs(full_score) < 1e-8:
        return 0.0
    return float(((full_score - truncated_score) / full_score) * 100.0)


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Patient-level bootstrap 95% confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
    }


def wilcoxon_paired_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test for continuous per-patient metrics (Dice, HD95)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    valid = (~np.isnan(a)) & (~np.isnan(b))
    diff = a[valid] - b[valid]
    if len(diff) < 3 or np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, pval = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(pval)}


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Paired McNemar test comparing two classifiers on identical held-out patients."""
    y = np.asarray(y_true)
    ca = np.asarray(pred_a) == y
    cb = np.asarray(pred_b) == y
    b = int(np.sum(ca & (~cb)))
    c = int(np.sum((~ca) & cb))
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(b + c)
    pval = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": pval, "a_only_correct": b, "b_only_correct": c}


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Dict[str, float]:
    """Fast DeLong variance test comparing two correlated ROC AUCs on identical test cases."""
    y = (np.asarray(y_true) > 0).astype(int)
    pa = np.asarray(prob_a, dtype=np.float64)
    pb = np.asarray(prob_b, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    m = int(np.sum(pos))
    n = int(np.sum(neg))
    if m < 2 or n < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "z_stat": 0.0, "p_value": 1.0}

    def _structural(scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        x = scores[pos][:, None]
        z = scores[neg][None, :]
        comp = (x > z).astype(np.float64) + 0.5 * (x == z).astype(np.float64)
        v10 = comp.mean(axis=1)
        v01 = comp.mean(axis=0)
        return float(v10.mean()), v10, v01

    auc_a, v10_a, v01_a = _structural(pa)
    auc_b, v10_b, v01_b = _structural(pb)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / m + s01 / n
    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var_diff <= 1e-12:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": 0.0, "p_value": 1.0}
    z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
    pval = float(2.0 * stats.norm.sf(abs(z_stat)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": float(z_stat), "p_value": pval}


def evaluate_xai_faithfulness(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    target_class: int,
    steps: int = 10,
) -> Dict[str, float]:
    """Compute quantitative XAI Deletion AUC and Insertion AUC faithfulness scores.

    Faithful explanations produce LOW Deletion AUC (confidence drops fast when top-salient
    pixels are masked) and HIGH Insertion AUC (confidence rises fast when salient pixels appear).
    """
    img = image.detach().clone()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    sal = saliency_map.detach().float()
    if sal.ndim == 3:
        sal = sal.squeeze(0)
    h, w = sal.shape
    order = torch.argsort(sal.flatten(), descending=True)
    n_pix = h * w

    del_scores = []
    ins_scores = []
    baseline = torch.zeros_like(img)

    for s in range(steps + 1):
        k = int(round((s / float(steps)) * n_pix))
        del_img = img.clone()
        ins_img = baseline.clone()
        if k > 0:
            top_idx = order[:k]
            ys = top_idx // w
            xs = top_idx % w
            del_img[:, :, ys, xs] = 0.0
            ins_img[:, :, ys, xs] = img[:, :, ys, xs]

        with torch.no_grad():
            p_del = float(predict_fn(del_img)[0, target_class].item())
            p_ins = float(predict_fn(ins_img)[0, target_class].item())
        del_scores.append(p_del)
        ins_scores.append(p_ins)

    x_axis = np.linspace(0.0, 1.0, steps + 1)
    return {
        "deletion_auc": float(np.trapezoid(del_scores, x_axis)),
        "insertion_auc": float(np.trapezoid(ins_scores, x_axis)),
    }


def export_latex_results_table(
    rows: List[Dict[str, str]],
    columns: List[str],
    caption: str,
    label: str,
    out_path: Optional[Path] = None,
) -> str:
    """Generate publication-ready booktabs LaTeX table directly from real experimental logs."""
    col_spec = "l" + "c" * (len(columns) - 1)
    header = " & ".join(f"\\textbf{{{c}}}" for c in columns) + " \\\\"
    body_lines = []
    for r in rows:
        body_lines.append(" & ".join(str(r.get(c, "--")) for c in columns) + " \\\\")
    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex, encoding="utf-8")
    return tex
