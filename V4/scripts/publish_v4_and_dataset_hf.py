#!/usr/bin/env python3
"""Publish PanOrgan 4 (Model + Card) and PanOrgan-Dataset (3.3 GB Shards + Manifest + Card) to Hugging Face."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/Users/theshoaibme/Desktop/Projects/ResearchPilot")
MODEL_SRC = ROOT / "V2" / "panorgan" / "models" / "pan_organ_net_v2.py"
CKPT_DIR = Path("/Volumes/Shoaib/PanOrgan_4/checkpoints/panorgan_4")
REPORT_JSON = Path("/Volumes/Shoaib/PanOrgan_4/experiments/thesis_reports/panorgan_4_complete_thesis_report.json")
SHARDS_DIR = Path("/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards")
STAGING_MODEL = Path("/Volumes/Shoaib/PanOrgan_HF_Staging/PanOrgan-4")


def build_v4_model_card(report: dict) -> str:
    ov = report["overall_classification"]
    seg = report["segmentation"]["overall"]
    mnar = report["mnar_stress_curve"]
    rows = []
    for hname, v in report["per_task_group"].items():
        rows.append(
            f"| `{hname}` | {v['n']} | {v['accuracy']*100:.2f}% | {v['macro_f1']:.4f} | {v['auroc']:.4f} | {v['ece']:.4f} | {v.get('decision_threshold', 0.5):.2f} |"
        )
    table_md = "\n".join(rows)

    return f"""---
license: mit
library_name: pytorch
tags:
  - medical-imaging
  - multi-modal
  - foundation-model
  - segmentation
  - classification
  - conformal-prediction
  - panorgan
  - pytorch
---

# PanOrgan 4 — Multi-Head & Multi-Scale Cohort-Isolated Foundation Model

**Project Title:** The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening (**Pan-Organ Net**)  
**Institution:** Daffodil International University — Department of Computer Science and Engineering (CSE499 FYDP, Spring 2026)  
**Dataset:** [`theshoaibme/PanOrgan-Dataset`](https://huggingface.co/datasets/theshoaibme/PanOrgan-Dataset)

## Overview
**PanOrgan 4** pairs a unified `PanOrganNetV2` backbone (`OmniModalityStem` + `PhysicsFiLM` + `ConvNeXt-Small` + `LoRA` + `Sparse Top-2 MoE` + `SACR`) with **cohort-isolated multi-scale spatial heads** (`32×32`, `16×16`, `8×8` Avg+Max pooled + `384-d` embedding = `2,688-d` with flip TTA) and per-cohort Class-Balanced Focal Loss, EMA (`0.998`), Temperature Scaling, and decision threshold tuning.

- **Model Weights:** `best.pt` (`243 MB`), `best_fp16.pt` (`122 MB`), `multiscale_heads.pt` (`46 MB`)
- **Dataset Scope:** `20,175` samples across `9,381` unique patients (`11` cohorts, `5` modalities: `CXR`, `MRI`, `CT`, `MG`, `US`, `0` patient-level leakage)

## Team & Supervision
- **1st Member:** MD. Shoaib Khan (ID: `0242310005101484`)
- **2nd Member:** Asmita Rahman (ID: `0242310005101412`)
- **Supervisor:** Dr. Md. Ali Hossain, Associate Professor, Dept. of CSE, Daffodil International University
- **Co-Supervisor:** Mayen Uddin Mojumder, Assistant Professor, Dept. of CSE, Daffodil International University

## Quantitative Test Results (`n = {ov['n_samples']}`)
- **Overall Classification Accuracy:** **`{ov['accuracy']*100:.2f}%`**
- **Overall Macro-F1:** **`{ov['macro_f1']:.4f}`** (Precision `{ov['precision_macro']:.4f}`, Recall `{ov['recall_macro']:.4f}`)
- **Segmentation Dice (MNAR 0% / 20% / 40%):** **`{seg['dice']['mean']:.4f}`** / **`{mnar['20']['overall']['dice']['mean']:.4f}`** / **`{mnar['40']['overall']['dice']['mean']:.4f}`** (`IoU = {seg['iou']['mean']:.4f}`)

### Per-Head Held-Out Test Breakdown
| Head / Cohort | Samples (`n`) | Accuracy | Macro-F1 | AUROC | ECE | Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
{table_md}

## Included Files
- `best.pt` — Main `PanOrgan 4` PyTorch checkpoint (`243 MB`)
- `best_fp16.pt` — Lightweight FP16 deployment checkpoint (`122 MB`)
- `multiscale_heads.pt` — Cohort-isolated multi-scale heads + per-cohort temperatures & thresholds (`46 MB`)
- `config.json` — Model metadata and cohort configuration
- `modeling_panorgan.py` — `PanOrganNetV2` architecture definition
- `thesis_report.json` — Full held-out evaluation report
"""


def build_dataset_card() -> str:
    return """---
license: mit
task_categories:
  - image-classification
  - image-segmentation
tags:
  - medical-imaging
  - multi-modal
  - chest-xray
  - mri
  - ct
  - mammography
  - ultrasound
  - panorgan
size_categories:
  - 10K<n<100K
---

# PanOrgan Multi-Modal Medical Screening Dataset (`PanOrgan-Dataset`)

**Project Title:** The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening (**Pan-Organ Net**)  
**Institution:** Daffodil International University — Department of Computer Science and Engineering (CSE499 FYDP, Spring 2026)  
**Model:** [`theshoaibme/PanOrgan-4`](https://huggingface.co/theshoaibme/PanOrgan-4)

## Dataset Summary
A unified, patient-level zero-leakage multi-modal medical imaging dataset stored as compressed `256×256` 2D/2.5D `.npz` shards (`3.3 GB` total) indexed by `master_manifest_v4.parquet`.

- **Total Samples:** `20,175`
- **Unique Patients:** `9,381` (`0` patient-level overlap across `train` / `val` / `test`)
- **Splits:** Train `13,707` | Val `3,241` | Test `3,227`
- **Modalities (5):**
  - `CXR`: `6,512` (`pulmonology_cxr`, `pulmonology_cxr_ext`, `nih_cxr14`)
  - `MRI`: `5,127` (`tcga_lgg`, `prostatex_mri`)
  - `CT`: `3,687` (`kidney_ct_abn`, `tcga_kirc_ct`)
  - `MG`: `2,696` (`cmmd_mg`, `cbis_ddsm`)
  - `US`: `2,153` (`busi`, `ultrasound_nerve`)

## Team & Supervision
- **1st Member:** MD. Shoaib Khan (ID: `0242310005101484`)
- **2nd Member:** Asmita Rahman (ID: `0242310005101412`)
- **Supervisor:** Dr. Md. Ali Hossain, Associate Professor, Dept. of CSE, Daffodil International University
- **Co-Supervisor:** Mayen Uddin Mojumder, Assistant Professor, Dept. of CSE, Daffodil International University
"""


def main():
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated on Hugging Face as: {user.get('name')} ({user.get('fullname')})")

    # 1. Stage & Publish PanOrgan-4 Model
    if STAGING_MODEL.exists():
        shutil.rmtree(STAGING_MODEL)
    STAGING_MODEL.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT_JSON.read_text())

    shutil.copy2(CKPT_DIR / "best.pt", STAGING_MODEL / "best.pt")
    shutil.copy2(CKPT_DIR / "best_fp16.pt", STAGING_MODEL / "best_fp16.pt")
    shutil.copy2(CKPT_DIR / "multiscale_heads.pt", STAGING_MODEL / "multiscale_heads.pt")
    shutil.copy2(MODEL_SRC, STAGING_MODEL / "modeling_panorgan.py")
    shutil.copy2(REPORT_JSON, STAGING_MODEL / "thesis_report.json")
    (STAGING_MODEL / "config.json").write_text(json.dumps({
        "model_name": "PanOrgan 4",
        "version": "4",
        "architecture": "PanOrganNetV2 + MultiScaleCohortHead",
        "repo_id": "theshoaibme/PanOrgan-4",
        "modalities": ["CXR", "MRI", "CT", "MG", "US"],
        "accuracy": report["overall_classification"]["accuracy"],
        "macro_f1": report["overall_classification"]["macro_f1"],
        "seg_dice": report["segmentation"]["overall"]["dice"]["mean"],
    }, indent=2))
    (STAGING_MODEL / "README.md").write_text(build_v4_model_card(report))

    model_repo = "theshoaibme/PanOrgan-4"
    print(f"\n[1/2] Publishing {model_repo}...")
    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(STAGING_MODEL),
        repo_id=model_repo,
        repo_type="model",
        commit_message="Publish PanOrgan 4 (best.pt, best_fp16.pt, multiscale_heads.pt, model card, report)",
    )
    print(f"Published Model -> https://huggingface.co/{model_repo}")

    # 2. Publish Single Unified Dataset (PanOrgan-Dataset, 3.3 GB)
    (SHARDS_DIR / "README.md").write_text(build_dataset_card())
    ds_repo = "theshoaibme/PanOrgan-Dataset"
    print(f"\n[2/2] Publishing {ds_repo} (3.3 GB shards + manifest)...")
    api.create_repo(repo_id=ds_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(SHARDS_DIR),
        repo_id=ds_repo,
        repo_type="dataset",
        commit_message="Publish unified PanOrgan-Dataset (20,175 samples, 9,381 patients, 3.3 GB shards + master_manifest_v4.parquet)",
    )
    print(f"Published Dataset -> https://huggingface.co/datasets/{ds_repo}")


if __name__ == "__main__":
    main()
