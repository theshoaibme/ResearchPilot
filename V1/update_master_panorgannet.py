import os
from pathlib import Path
from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parent

def update_master_panorgannet():
    print("=" * 65)
    print("🌐 Updating Master Unified PanOrganNet Dataset on Hugging Face")
    print("=" * 65)

    api = HfApi()
    repo_id = "theshoaibme/PanOrganNet"

    # 1. Generate Master Dataset Hub README.md
    readme_content = """---
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
---

# Pan-Organ Net: Complete Multi-Organ Medical Imaging Foundation Hub

**The Pan-Organ Diagnostic Paradigm (Pan-Organ Net)** establishes a high-capacity volumetric foundation model across diverse human organs (brain, lungs, liver, kidneys, prostate, breast, nerves) and imaging modalities (CT, MRI, Ultrasound, Mammography, and X-Ray).

---

## 🗂️ Unified Multi-Organ Dataset Portfolio

All datasets are available as modular sub-repositories and aggregated in this hub:

| Modality & Specialty | Dataset Name | Sub-Repository Link | Contents |
|---|---|---|---|
| **2D Ultrasound (Oncology)** | BUSI Breast Ultrasound | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-busi-breast-ultrasound) | Benign, malignant & normal breast lesions with ground-truth masks |
| **2D Chest X-Ray (Pulmonology)** | COVID-19 Radiography | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-pulmonology-chest-xray) | COVID-19, Normal, Lung Opacity, Viral Pneumonia |
| **2D Ultrasound (Neurology)** | Ultrasound Nerve | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-ultrasound-nerve-segmentation) | Brachial plexus nerve ultrasound with pixel mask annotations |
| **3D mpMRI (Urology/Oncology)** | PROSTATEx | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-prostatex-mri) | Multi-parametric prostate MRI series (T2w, DWI/ADC, DCE) |
| **3D Contrast CT & MRI (Nephrology)** | TCGA-KIRC | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-tcga-kirc-renal-ct-mri) | Clear cell renal cell carcinoma arterial & nephrographic series |
| **Full-Field Mammography (Oncology)** | CBIS-DDSM | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-cbis-ddsm-mammography) | Cranio-Caudal & Medio-Lateral Oblique screening mammograms |
| **Preprocessed Benchmark** | Processed Pulmonology | [🔗 View Dataset](https://huggingface.co/datasets/theshoaibme/panorgan-processed-pulmonology) | Standardized 299x299 radiographs with train/val/test splits & manifests |

---

## 💻 Quick Usage

### Download Entire Hub or Any Sub-Dataset:
```python
from huggingface_hub import snapshot_download

# Download any specific organ dataset:
busi_dir = snapshot_download(repo_id="theshoaibme/panorgan-busi-breast-ultrasound", repo_type="dataset")
cxr_dir = snapshot_download(repo_id="theshoaibme/panorgan-pulmonology-chest-xray", repo_type="dataset")
mri_dir = snapshot_download(repo_id="theshoaibme/panorgan-prostatex-mri", repo_type="dataset")
kirc_dir = snapshot_download(repo_id="theshoaibme/panorgan-tcga-kirc-renal-ct-mri", repo_type="dataset")
```

---

## 📖 Citation
```bibtex
@article{panorgan2026,
  title={The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening},
  author={The Shoaib and ResearchPilot Team},
  year={2026}
}
```
"""

    temp_readme = PROJECT_ROOT / "TEMP_MASTER_README.md"
    temp_readme.write_text(readme_content)

    print("Uploading master dataset card (README.md) to theshoaibme/PanOrganNet...")
    try:
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Add unified Pan-Organ Net Master Dataset Card & Index"
        )
        print("✓ Successfully published Master Dataset Card!")
    finally:
        if temp_readme.exists():
            temp_readme.unlink()

    # Upload CBIS_DDSM and TCGA_KIRC and PROSTATEx to master if desired
    for folder_name, local_sub in [
        ("CBIS_DDSM", PROJECT_ROOT / "dataset" / "raw" / "CBIS_DDSM"),
    ]:
        if local_sub.exists():
            print(f"\nUploading {folder_name} into {repo_id}...")
            try:
                api.upload_folder(
                    folder_path=str(local_sub),
                    path_in_repo=folder_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message=f"Add {folder_name} to PanOrganNet",
                    ignore_patterns=["*.DS_Store", "*.tmp"]
                )
                print(f"✓ Uploaded {folder_name}")
            except Exception as e:
                print(f"Notice: {e}")

    print("\n" + "=" * 65)
    print(f"🎉 Master Dataset Live at: https://huggingface.co/datasets/{repo_id}")
    print("=" * 65)

if __name__ == "__main__":
    update_master_panorgannet()
