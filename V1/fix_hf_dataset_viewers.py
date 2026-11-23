import os
import sys
from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm
from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parent

DICOM_DATASETS = [
    {
        "repo_name": "panorgan-cbis-ddsm-mammography",
        "local_dir": PROJECT_ROOT / "dataset" / "raw" / "CBIS_DDSM",
        "title": "Pan-Organ Net: CBIS-DDSM Digital Mammography Dataset",
        "specialty": "Radiology / Breast Oncology",
        "modality": "2D Digital Mammography",
        "task": "image-classification",
        "tags": ["medical", "mammography", "dicom", "breast-cancer", "oncology", "pan-organ-net"],
        "description": "Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) with microcalcifications and mass lesion DICOM series."
    },
    {
        "repo_name": "panorgan-prostatex-mri",
        "local_dir": PROJECT_ROOT / "dataset" / "raw" / "PROSTATEx",
        "title": "Pan-Organ Net: PROSTATEx Multi-Parametric MRI Dataset",
        "specialty": "Urology / Pelvic Oncology",
        "modality": "3D Multi-Parametric MRI",
        "task": "image-classification",
        "tags": ["medical", "mri", "dicom", "prostate-cancer", "urology", "oncology", "pan-organ-net"],
        "description": "Prostate Multi-Parametric MRI (T2-weighted, Diffusion-Weighted ADC/B-value, and Dynamic Contrast Enhanced DCE) from The Cancer Imaging Archive (TCIA)."
    },
    {
        "repo_name": "panorgan-tcga-kirc-renal-ct-mri",
        "local_dir": PROJECT_ROOT / "dataset" / "raw" / "TCGA_KIRC",
        "title": "Pan-Organ Net: TCGA-KIRC Renal CT & MRI Dataset",
        "specialty": "Nephrology / Urologic Oncology",
        "modality": "3D Contrast CT & MRI",
        "task": "image-classification",
        "tags": ["medical", "ct-scan", "mri", "dicom", "kidney-cancer", "nephrology", "tcia", "pan-organ-net"],
        "description": "Clear Cell Renal Cell Carcinoma (KIRC) contrast-enhanced CT and MRI scans from TCIA, representing renal tumors, cortex, and parenchymal tissue."
    }
]

def generate_dicom_metadata(local_dir: Path) -> pd.DataFrame:
    dcm_files = list(local_dir.glob("**/*.dcm"))
    print(f"Indexing {len(dcm_files)} DICOM files in {local_dir.name}...")
    
    rows = []
    for dcm in tqdm(dcm_files, desc=f"Parsing {local_dir.name}"):
        rel_path = str(dcm.relative_to(local_dir))
        file_size_kb = round(dcm.stat().st_size / 1024, 2)
        try:
            ds = pydicom.dcmread(dcm, stop_before_pixels=True)
            rows.append({
                "file_path": rel_path,
                "patient_id": str(getattr(ds, "PatientID", "Unknown")),
                "modality": str(getattr(ds, "Modality", "Unknown")),
                "series_description": str(getattr(ds, "SeriesDescription", "Unknown")),
                "body_part": str(getattr(ds, "BodyPartExamined", "Unknown")),
                "study_date": str(getattr(ds, "StudyDate", "Unknown")),
                "file_size_kb": file_size_kb
            })
        except Exception:
            rows.append({
                "file_path": rel_path,
                "patient_id": "Unknown",
                "modality": "Unknown",
                "series_description": "Unknown",
                "body_part": "Unknown",
                "study_date": "Unknown",
                "file_size_kb": file_size_kb
            })
    return pd.DataFrame(rows)

def make_readme_content(ds_cfg: dict, repo_id: str) -> str:
    tags_yaml = "\n".join([f"  - {t}" for t in ds_cfg["tags"]])
    return f"""---
license: cc-by-4.0
task_categories:
  - {ds_cfg['task']}
tags:
{tags_yaml}
pretty_name: "{ds_cfg['title']}"
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: "metadata.csv"
---

# {ds_cfg['title']}

Part of the **Pan-Organ Diagnostic Paradigm (Pan-Organ Net)** clinical foundation model research initiative.

## Overview
{ds_cfg['description']}

- **Medical Specialist**: {ds_cfg['specialty']}
- **Primary Imaging Modality**: {ds_cfg['modality']}
- **Clinical Task**: {ds_cfg['task']}

## Interactive Dataset Viewer
This repository includes a structured `metadata.csv` indexing all clinical series and DICOM files.
You can preview, search, filter, and inspect patient IDs, scanner modalities, anatomical target structures, and acquisition series directly in the Hugging Face Dataset Viewer above!

## Quick Usage with Hugging Face Hub

```python
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

# 1. Download and explore metadata table
meta_path = hf_hub_download(repo_id="{repo_id}", filename="metadata.csv", repo_type="dataset")
df = pd.read_csv(meta_path)
print(df.head())

# 2. Download raw clinical DICOM files
dataset_dir = snapshot_download(repo_id="{repo_id}", repo_type="dataset")
print(f"DICOM files downloaded to: {{dataset_dir}}")
```
"""

def fix_all_viewers():
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]

    print("=" * 65)
    print("🛠️ Enabling Hugging Face Dataset Viewers for All DICOM Repos")
    print("=" * 65)

    all_catalog_rows = []

    for ds in DICOM_DATASETS:
        repo_id = f"{username}/{ds['repo_name']}"
        local_dir = ds["local_dir"]
        print(f"\nProcessing: {repo_id}")

        if not local_dir.exists():
            print(f"Skipping {local_dir} (not found).")
            continue

        # 1. Generate metadata.csv
        df = generate_dicom_metadata(local_dir)
        csv_path = local_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(df)} records to {csv_path}")

        # Add to master catalog
        df_master = df.copy()
        df_master["dataset_name"] = ds["repo_name"]
        all_catalog_rows.append(df_master)

        # 2. Generate updated README.md with configs block
        readme_path = local_dir / "README.md"
        readme_path.write_text(make_readme_content(ds, repo_id))

        # 3. Upload metadata.csv and README.md
        print(f"Uploading metadata.csv and README.md to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo="metadata.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Add metadata.csv to enable interactive Dataset Viewer"
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Configure Dataset Viewer in dataset card"
        )
        print(f"🎉 Dataset Viewer enabled for: https://huggingface.co/datasets/{repo_id}")

    # Fix Master PanOrganNet
    print(f"\nProcessing Master: {username}/PanOrganNet...")
    if all_catalog_rows:
        master_df = pd.concat(all_catalog_rows, ignore_index=True)
        master_csv = PROJECT_ROOT / "dataset" / "raw" / "master_metadata.csv"
        master_df.to_csv(master_csv, index=False)

        master_readme = f"""---
license: cc-by-4.0
task_categories:
  - image-classification
  - image-segmentation
tags:
  - medical
  - foundation-model
  - multi-organ
  - multi-modality
  - mri
  - ct-scan
  - ultrasound
  - chest-xray
  - mammography
  - pan-organ-net
pretty_name: "Pan-Organ Net: Complete Multi-Modality Foundation Dataset Hub"
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: "metadata.csv"
---

# Pan-Organ Net: Complete Multi-Organ Medical Imaging Foundation Hub

**The Pan-Organ Diagnostic Paradigm (Pan-Organ Net)** establishes a high-capacity volumetric foundation model across diverse human organs (brain, lungs, liver, kidneys, prostate, breast, nerves) and imaging modalities (CT, MRI, Ultrasound, Mammography, and X-Ray).

## Interactive Dataset Viewer
Browse, search, and filter the unified multi-organ cohort below.

## Sub-Repositories Portfolio
- [panorgan-busi-breast-ultrasound](https://huggingface.co/datasets/theshoaibme/panorgan-busi-breast-ultrasound)
- [panorgan-pulmonology-chest-xray](https://huggingface.co/datasets/theshoaibme/panorgan-pulmonology-chest-xray)
- [panorgan-ultrasound-nerve-segmentation](https://huggingface.co/datasets/theshoaibme/panorgan-ultrasound-nerve-segmentation)
- [panorgan-prostatex-mri](https://huggingface.co/datasets/theshoaibme/panorgan-prostatex-mri)
- [panorgan-tcga-kirc-renal-ct-mri](https://huggingface.co/datasets/theshoaibme/panorgan-tcga-kirc-renal-ct-mri)
- [panorgan-cbis-ddsm-mammography](https://huggingface.co/datasets/theshoaibme/panorgan-cbis-ddsm-mammography)
- [panorgan-processed-pulmonology](https://huggingface.co/datasets/theshoaibme/panorgan-processed-pulmonology)
"""
        master_readme_file = PROJECT_ROOT / "TEMP_MASTER_VIEWER_README.md"
        master_readme_file.write_text(master_readme)

        api.upload_file(
            path_or_fileobj=str(master_csv),
            path_in_repo="metadata.csv",
            repo_id=f"{username}/PanOrganNet",
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Add unified metadata.csv to enable master Dataset Viewer"
        )
        api.upload_file(
            path_or_fileobj=str(master_readme_file),
            path_in_repo="README.md",
            repo_id=f"{username}/PanOrganNet",
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Configure master Dataset Viewer"
        )
        if master_readme_file.exists():
            master_readme_file.unlink()
        print("✓ Enabled Viewer for PanOrganNet Master Hub!")

    print("\n" + "=" * 65)
    print("✅ All Dataset Viewers Configured and Active!")
    print("=" * 65)

if __name__ == "__main__":
    fix_all_viewers()
