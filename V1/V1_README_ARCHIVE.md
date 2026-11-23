# V1 Archive — Phase-1 Exploratory Baselines

This directory preserves all files from the initial exploratory phase of `ResearchPilot` without deletion so every design evolution into `V2/` remains transparent and auditable.

## What Is Preserved Here
- `core/`: Phase-1 downloaders, per-dataset scripts, and isolated 2D CNN baselines (`DenseNet121`, `EfficientNetV2-S`).
- `dataset/`: Original locally downloaded datasets (`BUSI`, `Ultrasound_Nerve`, `PROSTATEx`, `TCGA_KIRC`, `CBIS_DDSM`, `pulmonology`).
- `docs/`, `reports/`, `plots/`, `research/`: Initial conceptual proposals and Phase-1 baseline evaluation plots.

## Why V2 Replaced V1
1. **Label Validity Audit:** Phase-1 preprocessing scripts for `TCGA-KIRC`, `PROSTATEx`, and `CBIS-DDSM` trained classifiers on scan-type/modality proxies rather than clinical pathology or expert segmentation masks.
2. **Unified Architecture:** Phase-1 contained only separate single-dataset 2D CNN classifiers rather than a unified multi-modality pan-organ network.
3. **Validation Rigor:** Phase-1 lacked external hospital validation cohorts, probability calibration, conformal uncertainty sets, and patient-level leakage assertions across all datasets.

All active research, corrected datasets, unified architecture code, and verified experimental results reside in [`../V2/`](../V2/).
