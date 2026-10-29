# PanOrganNet: Pulmonology Multi-Model Benchmark Test Report

**Test Cohort**: 3,175 independent holdout Chest X-Rays (Stratified 15% Split)
**Evaluation Date**: /Users/theshoaibme/Desktop/Projects/ResearchPilot

## Overall Comparative Performance

| Architecture | Parameters (M) | Test Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| **EfficientNetV2_S** | 20.18M | **95.81%** | 0.9741 | 0.9597 | **0.9663** | 0.9947 |
| **DenseNet121** | 6.96M | **94.27%** | 0.9487 | 0.9588 | **0.9535** | 0.9941 |

## Per-Class Sensitivity & Specificity Breakdown

| Architecture | Class | Sensitivity (Recall) | Specificity | Class F1 |
|---|---|---|---|---|
| EfficientNetV2_S | Normal | 98.36% | 93.56% | 0.9583 |
| EfficientNetV2_S | COVID | 98.52% | 99.92% | 0.9907 |
| EfficientNetV2_S | Lung_Opacity | 89.47% | 98.94% | 0.9313 |
| EfficientNetV2_S | Viral_Pneumonia | 97.52% | 99.97% | 0.9850 |
| DenseNet121 | Normal | 92.61% | 96.42% | 0.9427 |
| DenseNet121 | COVID | 98.34% | 99.39% | 0.9771 |
| DenseNet121 | Lung_Opacity | 93.57% | 95.56% | 0.9139 |
| DenseNet121 | Viral_Pneumonia | 99.01% | 99.80% | 0.9804 |