import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parent

def publish_processed_dataset():
    print("=" * 65)
    print("🚀 Publishing Processed Pulmonology & Splits Dataset to Hugging Face")
    print("=" * 65)

    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    repo_name = "panorgan-processed-pulmonology"
    repo_id = f"{username}/{repo_name}"

    processed_dir = PROJECT_ROOT / "dataset" / "processed" / "pulmonology" / "chest_xray"
    splits_dir = PROJECT_ROOT / "dataset" / "splits" / "pulmonology" / "chest_xray"
    archives_dir = PROJECT_ROOT / "dataset" / "raw" / "processed_archives"

    print(f"Target Repo: {repo_id}")
    print(f"Processed Dir: {processed_dir}")
    print(f"Splits Dir: {splits_dir}")

    # 1. Create or ensure repository exists
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
        private=False
    )
    print(f"✓ Repository ready: {url}")

    # 2. Upload README dataset card
    readme_content = f"""---
license: cc-by-4.0
task_categories:
  - image-classification
tags:
  - medical
  - chest-xray
  - pulmonology
  - covid-19
  - pneumonia
  - preprocessed
  - pan-organ-net
pretty_name: "Pan-Organ Net: Processed Pulmonology Dataset with Standard Splits"
size_categories:
  - 10K<n<100K
---

# Pan-Organ Net: Processed Pulmonology Dataset (with Train / Val / Test Splits)

Standardized, preprocessed production dataset for **Pan-Organ Net** multi-organ foundation model screening.

## Directory Layout & Schema
- `splits/`:
  - `train.csv`: 70% training cohort (14,815 images) with file paths and numeric class IDs
  - `validation.csv`: 15% validation cohort (3,175 images)
  - `test.csv`: 15% independent test benchmark (3,175 images)
- `data/`:
  - `train_images.tar.gz`: Preprocessed training image cohort (299x299 PNG)
  - `validation_images.tar.gz`: Preprocessed validation image cohort
  - `test_images.tar.gz`: Preprocessed test benchmark image cohort
- `labels/`: Standardized numeric and string label mappings (`label_map.json`)
- `manifests/`: Full SHA-256 and checksum integrity dataset manifests
- `metadata/`: Clinical study metadata
- `statistics/`: Class balance distributions and dataset telemetry

## Label Map
- `Normal`: 0
- `COVID`: 1
- `Lung_Opacity`: 2
- `Viral_Pneumonia`: 3

## Quick Usage

```python
import tarfile
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Download splits
train_csv_path = hf_hub_download(repo_id="{repo_id}", filename="splits/train.csv", repo_type="dataset")
train_df = pd.read_csv(train_csv_path)
print(train_df.head())

# 2. Download and extract test cohort
test_tar = hf_hub_download(repo_id="{repo_id}", filename="data/test_images.tar.gz", repo_type="dataset")
with tarfile.open(test_tar, "r:gz") as tar:
    tar.extractall("test_images")
```
"""
    temp_readme = PROJECT_ROOT / "TEMP_PROCESSED_README.md"
    temp_readme.write_text(readme_content)

    try:
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Add dataset card and schema for processed pulmonology"
        )
        print("✓ Uploaded README.md")
    finally:
        if temp_readme.exists():
            temp_readme.unlink()

    # 3. Upload splits folder into repo
    print("\nUploading splits (train.csv, validation.csv, test.csv)...")
    api.upload_folder(
        folder_path=str(splits_dir),
        path_in_repo="splits",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add official dataset train/validation/test splits"
    )
    print("✓ Uploaded splits folder")

    # 4. Upload labels, manifests, metadata, statistics
    for folder in ["labels", "manifests", "metadata", "statistics"]:
        f_path = processed_dir / folder
        if f_path.exists():
            print(f"Uploading {folder}...")
            api.upload_folder(
                folder_path=str(f_path),
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Add {folder}"
            )
            print(f"✓ Uploaded {folder}")

    # 5. Upload image split archives
    print("\nUploading data archives (train_images.tar.gz, validation_images.tar.gz, test_images.tar.gz)...")
    api.upload_folder(
        folder_path=str(archives_dir),
        path_in_repo="data",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add partitioned preprocessed image archives"
    )
    print(f"\n🎉 SUCCESS: Published {repo_id} -> https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    publish_processed_dataset()
