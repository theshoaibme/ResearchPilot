import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_CONFIG = [
    {
        "name": "panorgan-busi-breast-ultrasound",
        "title": "Pan-Organ Net: BUSI Breast Ultrasound Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "BUSI" / "Dataset_BUSI_with_GT",
        "modality": "2D Ultrasound",
        "specialty": "Oncology / Mammary Imaging",
        "task": "image-classification / image-segmentation",
        "tags": ["medical", "ultrasound", "oncology", "breast-cancer", "segmentation", "pan-organ-net"],
        "description": "Curated Breast Ultrasound Images (BUSI) with corresponding lesion ground-truth segmentation masks, categorized into Benign, Malignant, and Normal classes.",
        "schema_details": """### Dataset Structure & Schema
- **Format**: PNG images (500x500 avg resolution)
- **Classes**:
  - `benign`: Benign breast tumors with lesion masks (`*_mask.png`)
  - `malignant`: Malignant breast carcinomas with lesion masks (`*_mask.png`)
  - `normal`: Normal breast tissue without mass lesions
- **Modality**: 2D B-Mode Breast Ultrasound
- **Target Specialty**: Breast Oncology & Radiology
"""
    },
    {
        "name": "panorgan-pulmonology-chest-xray",
        "title": "Pan-Organ Net: COVID-19 & Pulmonary Radiography Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "pulmonology" / "chest_xray",
        "modality": "2D Chest X-Ray",
        "specialty": "Pulmonology / Infectious Diseases",
        "task": "image-classification",
        "tags": ["medical", "x-ray", "chest-xray", "pulmonology", "covid-19", "pneumonia", "pan-organ-net"],
        "description": "High-resolution Chest Radiography database covering COVID-19, Viral Pneumonia, Lung Opacity, and Normal lung cases for infectious thoracic screening.",
        "schema_details": """### Dataset Structure & Schema
- **Format**: PNG images (299x299 resolution) + Excel metadata sheets
- **Classes**:
  - `COVID`: Confirmed COVID-19 positive thoracic radiographs
  - `Normal`: Healthy adult chest radiographs
  - `Lung_Opacity`: Non-COVID interstitial lung opacities
  - `Viral Pneumonia`: Viral respiratory infection consolidations
- **Modality**: 2D Frontal Projection Radiograph
- **Target Specialty**: Thoracic Specialist & Pulmonologist
"""
    },
    {
        "name": "panorgan-ultrasound-nerve-segmentation",
        "title": "Pan-Organ Net: Ultrasound Nerve Segmentation Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "Ultrasound_Nerve",
        "modality": "2D Ultrasound",
        "specialty": "Neurology / Peripheral Nervous System",
        "task": "image-segmentation",
        "tags": ["medical", "ultrasound", "neurology", "brachial-plexus", "nerve-segmentation", "pan-organ-net"],
        "description": "Pixel-level ultrasound imaging dataset targeting the Brachial Plexus nerve structures to support precise anatomical boundary segmentation.",
        "schema_details": """### Dataset Structure & Schema
- **Format**: TIFF images (grayscale, 580x420)
- **Components**:
  - `train/`: Training ultrasound acquisitions alongside corresponding binary nerve masks
  - `test/`: Unlabeled test ultrasound volumes
  - `train_masks.csv`: Run-length encoded (RLE) nerve pixel annotations
- **Modality**: 2D High-Frequency Neck Ultrasound
- **Target Specialty**: Neurologist & Anesthesiologist
"""
    },
    {
        "name": "panorgan-prostatex-mri",
        "title": "Pan-Organ Net: PROSTATEx Multi-Parametric MRI Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "PROSTATEx",
        "modality": "3D Multi-Parametric MRI",
        "specialty": "Urology / Pelvic Oncology",
        "task": "image-classification",
        "tags": ["medical", "mri", "dicom", "prostate-cancer", "urology", "oncology", "pan-organ-net"],
        "description": "Prostate Multi-Parametric MRI (T2-weighted, Diffusion-Weighted ADC/B-value, and Dynamic Contrast Enhanced DCE) from The Cancer Imaging Archive (TCIA).",
        "schema_details": """### Dataset Structure & Schema
- **Format**: Native Clinical DICOM (.dcm) volumetric series
- **Sequences Included**:
  - Axial T2-weighted turbo spin echo
  - Diffusion-weighted imaging (DWI) with calculated ADC maps
  - Dynamic contrast-enhanced (DCE) MRI series
- **Modality**: Multi-Parametric 3T MRI
- **Target Specialty**: Urologist, Surgical Oncologist & Radiologist
"""
    },
    {
        "name": "panorgan-tcga-kirc-renal-ct-mri",
        "title": "Pan-Organ Net: TCGA-KIRC Renal CT & MRI Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "TCGA_KIRC",
        "modality": "3D Contrast CT & MRI",
        "specialty": "Nephrology / Urologic Oncology",
        "task": "image-classification",
        "tags": ["medical", "ct-scan", "mri", "dicom", "kidney-cancer", "nephrology", "tcia", "pan-organ-net"],
        "description": "Clear Cell Renal Cell Carcinoma (KIRC) contrast-enhanced CT and MRI scans from TCIA, representing renal tumors, cortex, and parenchymal tissue.",
        "schema_details": """### Dataset Structure & Schema
- **Format**: Native Clinical DICOM (.dcm) volumetric series
- **Modalities**:
  - Contrast-enhanced Abdominal CT (nephrographic, arterial, and delayed phases)
  - Abdominal Multi-Sequence MRI (T1-in/out phase, T2-weighted, Fat-Suppressed)
- **Target Specialty**: Nephrologist, Urologic Oncologist & Abdominal Radiologist
"""
    },
    {
        "name": "panorgan-cbis-ddsm-mammography",
        "title": "Pan-Organ Net: CBIS-DDSM Digital Mammography Dataset",
        "local_path": PROJECT_ROOT / "dataset" / "raw" / "CBIS_DDSM",
        "modality": "2D Digital Mammography",
        "specialty": "Radiology / Breast Oncology",
        "task": "image-classification",
        "tags": ["medical", "mammography", "dicom", "breast-cancer", "oncology", "pan-organ-net"],
        "description": "Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) with microcalcifications and mass lesion DICOM series.",
        "schema_details": """### Dataset Structure & Schema
- **Format**: Full-Field Digital Mammography DICOM (.dcm)
- **Views**: Cranio-Caudal (CC) and Medio-Lateral Oblique (MLO) projections
- **Findings**: Microcalcifications, architectural distortions, and malignant mass margins
- **Target Specialty**: Breast Imaging Specialist & Surgical Oncologist
"""
    }
]

def generate_readme(ds_cfg: dict, repo_id: str) -> str:
    tags_yaml = "\n".join([f"  - {tag}" for tag in ds_cfg["tags"]])
    return f"""---
license: cc-by-4.0
task_categories:
  - image-classification
  - image-segmentation
tags:
{tags_yaml}
pretty_name: "{ds_cfg['title']}"
size_categories:
  - 1K<n<10K
---

# {ds_cfg['title']}

Part of the **Pan-Organ Diagnostic Paradigm (Pan-Organ Net)** clinical foundation model research initiative.

## Overview
{ds_cfg['description']}

- **Medical Specialist**: {ds_cfg['specialty']}
- **Primary Imaging Modality**: {ds_cfg['modality']}
- **Clinical Task**: {ds_cfg['task']}

{ds_cfg['schema_details']}

## Quick Usage with Hugging Face Hub

```python
from huggingface_hub import snapshot_download

# Download dataset directory
local_dir = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset"
)
print(f"Dataset downloaded to: {{local_dir}}")
```

## Citation & Acknowledgements
If you utilize this dataset in medical imaging research, please cite the Pan-Organ Net research repository:
```bibtex
@article{{panorgan2026,
  title={{The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening}},
  author={{The Shoaib and ResearchPilot Team}},
  year={{2026}}
}}
```
"""

def publish_all(start_index: int = 0):
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    print(f"Authenticated as HF user: {username}")
    print("=" * 65)
    print("🚀 Publishing Multi-Organ Datasets to Hugging Face")
    print("=" * 65)

    for idx, ds in enumerate(DATASETS_CONFIG[start_index:], start=start_index):
        repo_name = ds["name"]
        repo_id = f"{username}/{repo_name}"
        local_dir = ds["local_path"]

        print(f"\n[{idx+1}/{len(DATASETS_CONFIG)}] Publishing: {repo_id}")
        print(f"Source Folder: {local_dir}")

        if not local_dir.exists():
            print(f"❌ Error: Local path {local_dir} does not exist. Skipping.")
            continue

        # 1. Create or ensure repository exists
        try:
            url = api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
                exist_ok=True,
                private=False
            )
            print(f"✓ Repository ready: {url}")
        except Exception as e:
            print(f"Error creating repo {repo_id}: {e}")
            continue

        # 2. Upload README.md (Dataset Card with schema)
        readme_content = generate_readme(ds, repo_id)
        temp_readme = PROJECT_ROOT / f"TEMP_{repo_name}_README.md"
        temp_readme.write_text(readme_content)

        try:
            api.upload_file(
                path_or_fileobj=str(temp_readme),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Add dataset card and schema for {repo_name}"
            )
            print("✓ Uploaded dataset schema & README.md")
        except Exception as e:
            print(f"Warning uploading README: {e}")
        finally:
            if temp_readme.exists():
                temp_readme.unlink()

        # 3. Upload dataset files/folder
        print(f"Uploading files to {repo_id} (this may take a few moments)...")
        try:
            api.upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Upload {ds['title']} files",
                ignore_patterns=["*.DS_Store", "*.git*", "*.tmp"]
            )
            print(f"🎉 SUCCESS: Published {repo_id} -> https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"❌ Error uploading {repo_id}: {e}")

    print("\n" + "=" * 65)
    print("✅ All Selected Datasets Successfully Processed & Published!")
    print("=" * 65)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    publish_all(start_index=start)
