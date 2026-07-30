import os
import sys
import getpass
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_BASE = BASE_DIR / "dataset"

def download_physionet_datasets():
    print("==================================================")
    print("PhysioNet Downloader (MIMIC-CXR & VinDr-CXR)")
    print("==================================================")

    user = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")

    if not user:
        user = input("PhysioNet Username: ").strip()
    if not password:
        password = getpass.getpass("PhysioNet Password: ").strip()

    if not user or not password:
        print("Error: Username and password are required for PhysioNet access.")
        sys.exit(1)

    datasets = [
        {"name": "MIMIC-CXR-JPG", "url": "https://physionet.org/files/mimic-cxr-jpg/2.0.0/", "dir": "MIMIC_CXR"},
        {"name": "VinDr-CXR", "url": "https://physionet.org/files/vindr-cxr/1.0.0/", "dir": "VinDr_CXR"}
    ]

    for ds in datasets:
        target_path = DATASET_BASE / ds["dir"]
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {ds['name']} to {target_path}...")

        cmd = [
            "wget", "-r", "-N", "-c", "-np",
            f"--user={user}",
            f"--password={password}",
            "-P", str(target_path),
            ds["url"]
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"SUCCESS: Completed {ds['name']}\n")
        except Exception as e:
            print(f"ERROR downloading {ds['name']}: {e}\n")

if __name__ == "__main__":
    download_physionet_datasets()
