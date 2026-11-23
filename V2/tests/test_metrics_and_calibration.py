"""Tests for clinical evaluation metrics, statistical tests, calibration, MC2 conformal sets, and XAI."""

from pathlib import Path

import numpy as np
import torch

from panorgan.evaluation import (
    ModalityConditionalConformal,
    TemperatureScaler,
    bootstrap_confidence_interval,
    compute_classification_metrics,
    compute_dice_and_hd95,
    compute_ece_and_brier,
    compute_mnar_degradation,
    delong_roc_test,
    evaluate_xai_faithfulness,
    export_latex_results_table,
    mcnemar_test,
    wilcoxon_paired_test,
)


def test_dice_and_physical_mm_hd95_2d_and_3d():
    true_2d = np.zeros((32, 32), dtype=np.uint8)
    true_2d[8:24, 8:24] = 1
    pred_2d = np.zeros((32, 32), dtype=np.uint8)
    pred_2d[8:24, 10:26] = 1

    res = compute_dice_and_hd95(pred_2d, true_2d, spacing_mm=(1.5, 1.5))
    assert 0.8 < res["dice"] < 0.95
    assert abs(res["hd95_mm"] - 3.0) < 0.5  # 2 pixels * 1.5 mm = 3.0 mm

    # Perfect 3D match
    vol = np.zeros((8, 16, 16), dtype=np.uint8)
    vol[2:6, 4:12, 4:12] = 1
    res3d = compute_dice_and_hd95(vol, vol, spacing_mm=(2.0, 1.0, 1.0))
    assert res3d["dice"] == 1.0 and res3d["hd95_mm"] == 0.0


def test_classification_bootstrap_and_statistical_tests():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 100 + [1] * 100)
    prob_strong = np.concatenate([rng.uniform(0.05, 0.45, 100), rng.uniform(0.55, 0.95, 100)])
    prob_weak = np.concatenate([rng.uniform(0.15, 0.65, 100), rng.uniform(0.35, 0.85, 100)])

    metrics = compute_classification_metrics(y_true, prob_strong)
    assert metrics["auroc"] > 0.95
    assert metrics["sens_at_95spec"] >= 0.80

    dl = delong_roc_test(y_true, prob_strong, prob_weak)
    assert dl["auc_a"] > dl["auc_b"]
    assert dl["p_value"] < 0.01

    mc = mcnemar_test(y_true, (prob_strong >= 0.5).astype(int), (prob_weak >= 0.5).astype(int))
    assert mc["p_value"] < 0.05

    w = wilcoxon_paired_test(np.linspace(0.85, 0.95, 20), np.linspace(0.75, 0.85, 20))
    assert w["p_value"] < 0.001

    ci = bootstrap_confidence_interval(np.linspace(0.80, 0.90, 50), n_bootstrap=500)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert abs(compute_mnar_degradation(0.90, 0.855) - 5.0) < 1e-4


def test_temperature_scaling_and_mc2_conformal_coverage():
    rng = np.random.default_rng(123)
    n_cal, n_test = 300, 400
    cal_logits = torch.randn(n_cal, 3) * 3.0
    cal_probs_true = torch.softmax(cal_logits / 2.5, dim=-1).numpy()
    cal_labels = np.array([rng.choice(3, p=p) for p in cal_probs_true])

    scaler = TemperatureScaler()
    temp = scaler.fit(cal_logits, torch.from_numpy(cal_labels))
    assert temp > 1.0  # Uncalibrated sharp logits require T > 1

    raw_ece = compute_ece_and_brier(cal_labels, torch.softmax(cal_logits, dim=-1).numpy())["ece"]
    cal_ece = compute_ece_and_brier(cal_labels, scaler.transform_probs(cal_logits))["ece"]
    assert cal_ece <= raw_ece

    # Modality-Conditional Conformal Prediction (MC2) across 2 modality patterns
    cal_groups = ["CT_full"] * 150 + ["US_partial"] * 150
    test_logits = torch.randn(n_test, 3) * 3.0
    test_probs_true = torch.softmax(test_logits / 2.5, dim=-1).numpy()
    test_labels = np.array([rng.choice(3, p=p) for p in test_probs_true])
    test_groups = ["CT_full"] * 200 + ["US_partial"] * 200

    mc2 = ModalityConditionalConformal(alpha=0.10)
    mc2.fit(scaler.transform_probs(cal_logits), cal_labels, cal_groups)
    cov = mc2.evaluate_coverage(scaler.transform_probs(test_logits), test_labels, test_groups)
    # Finite-sample empirical coverage should be near or above target 0.90
    assert cov["marginal_coverage"] >= 0.84
    assert "CT_full" in cov["per_group"] and "US_partial" in cov["per_group"]


def test_xai_faithfulness_and_latex_export(tmp_path: Path):
    # Toy model that directly depends on top-left corner intensity
    def predict_fn(x: torch.Tensor) -> torch.Tensor:
        score = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        return torch.stack([1.0 - score, score], dim=-1)

    img = torch.zeros(1, 3, 16, 16)
    img[:, :, :8, :8] = 1.0
    sal = torch.zeros(16, 16)
    sal[:8, :8] = 1.0

    faith = evaluate_xai_faithfulness(predict_fn, img, sal, target_class=1, steps=8)
    assert faith["insertion_auc"] > faith["deletion_auc"]

    tex = export_latex_results_table(
        rows=[{"Model": "Pan-Organ Net V2", "Dice": "0.892 (0.881--0.903)"}],
        columns=["Model", "Dice"],
        caption="Benchmark Results",
        label="tab:bench",
        out_path=tmp_path / "table.tex",
    )
    assert "\\toprule" in tex and (tmp_path / "table.tex").exists()
