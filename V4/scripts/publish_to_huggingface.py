#!/usr/bin/env python3
"""Publish all completed PanOrgan models, model cards, configs, code, JSON reports, and plots to Hugging Face Hub."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/Users/theshoaibme/Desktop/Projects/ResearchPilot")
ASSETS_DIR = ROOT / "reports" / "assets"
MODEL_SRC = ROOT / "V2" / "panorgan" / "models" / "pan_organ_net_v2.py"
STAGING_ROOT = Path("/Volumes/Shoaib/PanOrgan_HF_Staging")

MODELS = [
    {
        "version": "2",
        "repo_id": "theshoaibme/PanOrgan-2",
        "title": "PanOrgan 2",
        "subtitle": "Multi-Cohort Joint Foundation Model (3 Core Cohorts)",
        "checkpoint": Path("/Volumes/Shoaib/PanOrgan_V2/checkpoints/stage2_panorgan_v2_full/best.pt"),
        "report_json": Path("/Volumes/Shoaib/PanOrgan_V2/experiments/thesis_reports/complete_thesis_report.json"),
        "asset_prefix": "v2_",
        "modalities": ["MRI", "CT", "Ultrasound (US)"],
        "cohorts": ["prostatex_mri", "kidney_ct_mri", "ultrasound_nerve"],
        "highlights": "First unified multi-cohort PanOrganNetV2 release trained on Apple M4 MPS with MA-UGM self-supervised pretraining, temperature scaling, and MC2 conformal prediction.",
    },
    {
        "version": "3",
        "repo_id": "theshoaibme/PanOrgan-3",
        "title": "PanOrgan 3",
        "subtitle": "4-Modality Calibrated Foundation Model (Zero-Leakage)",
        "checkpoint": Path("/Volumes/Shoaib/PanOrgan_V3/checkpoints/v3_stage2b_calibrated/best.pt"),
        "report_json": Path("/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v3_complete_thesis_report.json"),
        "asset_prefix": "v3_",
        "modalities": ["MRI", "CT", "Ultrasound (US)", "Chest X-Ray (CXR)"],
        "cohorts": ["pulmonary_cxr", "busi_breast_us", "kidney_ct_mri", "ultrasound_nerve", "prostatex_mri"],
        "highlights": "Expanded zero-leakage manifest across 5 cohorts with Stage 1 SSL, Stage 2a joint training, Stage 2b temperature-scaled calibration, and MNAR missingness stress evaluation.",
    },
    {
        "version": "3.1",
        "repo_id": "theshoaibme/PanOrgan-3.1",
        "title": "PanOrgan 3.1",
        "subtitle": "5-Modality MixUp Foundation Model (8 Cohorts, 8,213 Patients)",
        "checkpoint": Path("/Volumes/Shoaib/PanOrgan_V3/checkpoints/v31_5modal_mixup/best.pt"),
        "report_json": Path("/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v31_complete_thesis_report.json"),
        "asset_prefix": "v31_",
        "modalities": ["MRI", "CT", "Ultrasound (US)", "Chest X-Ray (CXR)", "Mammography (MG)"],
        "cohorts": [
            "pulmonary_cxr", "busi_breast_us", "kidney_ct_mri", "ultrasound_nerve",
            "vindr_mammo_mg", "tcga_kirc_ct", "cbis_ddsm", "prostatex_mri"
        ],
        "highlights": "Adds Mammography (MG) and Renal CT cohorts with cross-modality MixUp regularization across 14,915 samples and 8,213 unique patients.",
    },
    {
        "version": "3.2",
        "repo_id": "theshoaibme/PanOrgan-3.2",
        "title": "PanOrgan 3.2",
        "subtitle": "Balanced 5-Modality Foundation Model (11 Cohorts, 20,175 Samples)",
        "checkpoint": Path("/Volumes/Shoaib/PanOrgan_V3/checkpoints/v32_balanced/best.pt"),
        "report_json": Path("/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v32_complete_thesis_report.json"),
        "asset_prefix": "v32_",
        "modalities": ["MRI", "CT", "Ultrasound (US)", "Chest X-Ray (CXR)", "Mammography (MG)"],
        "cohorts": [
            "pulmonary_cxr", "cxr_multiclass", "busi_breast_us", "ultrasound_nerve",
            "kidney_ct_mri", "kits23_ct", "tcga_kirc_ct", "cbis_ddsm",
            "vindr_mammo_mg", "cmmd_mg", "prostatex_mri"
        ],
        "highlights": "Balanced multi-cohort release integrating KiTS23, CMMD, and multi-class CXR across 20,175 samples and 9,381 unique patients with 0 patient-level leakage.",
    },
]


def fmt(val, digits=4):
    if isinstance(val, (int, float)):
        return f"{val:.{digits}f}"
    return str(val)


def build_metrics_markdown(version: str, report: dict) -> str:
    lines = []
    if version == "2":
        v2 = report.get("PanOrganNet_V2_Full", {})
        seg = v2.get("segmentation", {})
        cls_cal = v2.get("classification_calibrated", {})
        cls_unc = v2.get("classification_uncalibrated", {})
        mc2 = v2.get("mc2_conformal_coverage", {})
        if seg:
            lines.append("### Segmentation Performance (Held-Out Test)")
            lines.append("| Cohort | Samples | Mean Dice | IoU | HD95 (mm) |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for k, v in seg.items():
                lines.append(f"| `{k}` | {v.get('n', '-')} | {fmt(v.get('dice_mean'))} | {fmt(v.get('iou_mean'))} | {fmt(v.get('hd95_mean'), 2)} |")
            lines.append("")
        if cls_cal or cls_unc:
            lines.append("### Classification & Calibration")
            lines.append("| Metric | Uncalibrated | Calibrated |")
            lines.append("| --- | ---: | ---: |")
            for m in ["auroc_macro", "macro_f1", "accuracy", "ece", "brier"]:
                if m in cls_cal or m in cls_unc:
                    lines.append(f"| `{m}` | {fmt(cls_unc.get(m, '-'))} | {fmt(cls_cal.get(m, '-'))} |")
            lines.append("")
        if mc2:
            lines.append("### MC2 Conformal Coverage (alpha = 0.10)")
            lines.append("| Cohort | Test n | Marginal Coverage | Avg Set Size | Abstention Rate |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for k, v in mc2.items():
                if isinstance(v, dict):
                    lines.append(f"| `{k}` | {v.get('n', '-')} | {fmt(v.get('marginal_coverage', v.get('coverage')))} | {fmt(v.get('avg_set_size'), 2)} | {fmt(v.get('abstention_rate'))} |")
            lines.append("")
    else:
        seg = report.get("segmentation", {})
        cls_cal = report.get("classification_calibrated", {})
        cls_unc = report.get("classification_uncalibrated", {})
        mc2 = report.get("mc2_conformal_coverage", {})
        temp = report.get("optimal_temperature")

        if cls_cal or cls_unc:
            lines.append("### Classification & Calibration (Held-Out Test)")
            if temp is not None:
                lines.append(f"- **Optimal Temperature (`T`):** `{fmt(temp, 4)}`")
            lines.append("")
            lines.append("| Metric | Uncalibrated | Calibrated |")
            lines.append("| --- | ---: | ---: |")
            for m in ["auroc_macro", "macro_f1", "accuracy", "ece", "brier", "n"]:
                if m in cls_cal or m in cls_unc:
                    u = cls_unc.get(m, "-")
                    c = cls_cal.get(m, "-")
                    lines.append(f"| `{m}` | {u if m == 'n' else fmt(u)} | {c if m == 'n' else fmt(c)} |")
            lines.append("")

        if seg:
            lines.append("### Segmentation Performance (Held-Out Test)")
            lines.append("| Cohort | Samples | Mean Dice | IoU | HD95 (mm) |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for k, v in seg.items():
                if isinstance(v, dict):
                    lines.append(f"| `{k}` | {v.get('n', '-')} | {fmt(v.get('dice_mean'))} | {fmt(v.get('iou_mean'))} | {fmt(v.get('hd95_mean'), 2)} |")
            lines.append("")

        if mc2:
            lines.append("### MC2 Conformal Prediction Coverage (alpha = 0.10)")
            lines.append("| Cohort | Test n | Marginal Coverage | Avg Set Size | Abstention Rate |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for k, v in mc2.items():
                if isinstance(v, dict):
                    cov = v.get("marginal_coverage", v.get("coverage"))
                    lines.append(f"| `{k}` | {v.get('n', '-')} | {fmt(cov)} | {fmt(v.get('avg_set_size'), 2)} | {fmt(v.get('abstention_rate'))} |")
            lines.append("")
    return "\n".join(lines)


def build_model_card(spec: dict, report: dict, figure_names: list[str]) -> str:
    metrics_md = build_metrics_markdown(spec["version"], report)
    figs_md = "\n".join(f"- ![{name}](figures/{name})" for name in figure_names)
    modalities_str = ", ".join(spec["modalities"])
    cohorts_str = ", ".join(f"`{c}`" for c in spec["cohorts"])

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

# {spec['title']} — {spec['subtitle']}

**Project Title:** The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening (**Pan-Organ Net**)  
**Institution:** Daffodil International University — Department of Computer Science and Engineering (CSE499 FYDP, Spring 2026)

## Overview
{spec['highlights']}

- **Supported Modalities:** {modalities_str}
- **Training Cohorts ({len(spec['cohorts'])}):** {cohorts_str}
- **Backbone:** `PanOrganNetV2` (defined in `modeling_panorgan.py`)
- **Checkpoint:** `best.pt`
- **Full Evaluation JSON:** `thesis_report.json`

## Team & Supervision
- **1st Member:** MD. Shoaib Khan (ID: `0242310005101484`)
- **2nd Member:** Asmita Rahman (ID: `0242310005101412`)
- **Supervisor:** Dr. Md. Ali Hossain, Associate Professor, Dept. of CSE, Daffodil International University
- **Co-Supervisor:** Mayen Uddin Mojumder, Assistant Professor, Dept. of CSE, Daffodil International University

## Quantitative Evaluation Results (from `thesis_report.json`)

{metrics_md}

## Included Files
- `best.pt` — Trained PyTorch checkpoint weights
- `config.json` — Model configuration, modality list, and release metadata
- `modeling_panorgan.py` — `PanOrganNetV2` PyTorch model definition
- `thesis_report.json` — Full held-out test metrics, conformal coverage, and calibration data
- `figures/` — Diagnostic and evaluation plots ({len(figure_names)} PNG figures)

## Evaluation Figures
{figs_md}

## Quick Usage (PyTorch)
```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="{spec['repo_id']}", filename="best.pt")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("Loaded keys:", list(state.keys()) if isinstance(state, dict) else type(state))
```
"""


def main():
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated on Hugging Face as: {user.get('name')} ({user.get('fullname')})")

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in MODELS:
        repo_id = spec["repo_id"]
        stage_dir = STAGING_ROOT / repo_id.split("/")[-1]
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = stage_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        assert spec["checkpoint"].exists(), f"Missing checkpoint: {spec['checkpoint']}"
        assert spec["report_json"].exists(), f"Missing report JSON: {spec['report_json']}"

        report = json.loads(spec["report_json"].read_text())

        # Copy checkpoint, modeling code, JSON report, and config.json
        shutil.copy2(spec["checkpoint"], stage_dir / "best.pt")
        shutil.copy2(MODEL_SRC, stage_dir / "modeling_panorgan.py")
        shutil.copy2(spec["report_json"], stage_dir / "thesis_report.json")

        config_obj = {
            "model_name": spec["title"],
            "version": spec["version"],
            "architecture": "PanOrganNetV2",
            "repo_id": repo_id,
            "modalities": spec["modalities"],
            "cohorts": spec["cohorts"],
            "optimal_temperature": report.get("optimal_temperature"),
            "authors": ["MD. Shoaib Khan", "Asmita Rahman"],
            "supervisors": ["Dr. Md. Ali Hossain", "Mayen Uddin Mojumder"],
            "institution": "Daffodil International University (Dept. of CSE)",
        }
        (stage_dir / "config.json").write_text(json.dumps(config_obj, indent=2))

        # Copy matching PNG figures
        figure_names = []
        for png in sorted(ASSETS_DIR.glob(f"{spec['asset_prefix']}*.png")):
            shutil.copy2(png, fig_dir / png.name)
            figure_names.append(png.name)

        # Write Model Card README.md
        readme = build_model_card(spec, report, figure_names)
        (stage_dir / "README.md").write_text(readme)

        print(f"\n[{spec['title']}] Creating/updating repo {repo_id} (figures={len(figure_names)})...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {spec['title']} checkpoint, model card, config, modeling code, report JSON, and evaluation figures",
        )
        print(f"[{spec['title']}] Published -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
