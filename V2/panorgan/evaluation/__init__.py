"""Clinical evaluation metrics, statistical testing, calibration, MC2 conformal sets, and XAI."""

from panorgan.evaluation.metrics_and_calibration import (
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

__all__ = [
    "ModalityConditionalConformal",
    "TemperatureScaler",
    "bootstrap_confidence_interval",
    "compute_classification_metrics",
    "compute_dice_and_hd95",
    "compute_ece_and_brier",
    "compute_mnar_degradation",
    "delong_roc_test",
    "evaluate_xai_faithfulness",
    "export_latex_results_table",
    "mcnemar_test",
    "wilcoxon_paired_test",
]
