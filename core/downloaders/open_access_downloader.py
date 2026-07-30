import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_BASE = BASE_DIR / "dataset"

def download_totalsegmentator():
    print("==================================================")
    print("Open-Access Downloader: TotalSegmentator v2 (Zenodo)")
    print("==================================================")

    target_dir = DATASET_BASE / "TotalSegmentator"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "Totalsegmentator_dataset_v201.zip"
    url = "https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content"

    print(f"Target Directory: {target_dir}")
    if not zip_path.exists():
        cmd = f"curl -C - -L '{url}' -o '{zip_path}'"
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True)
    else:
        print(f"File already exists: {zip_path}")

if __name__ == "__main__":
    download_totalsegmentator()
