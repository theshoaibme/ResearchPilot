import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "raw" / "PanOrganNet"

def download_hf_dataset(token=None):
    if not token:
        token = os.environ.get("HF_TOKEN", HF_TOKEN)

    dataset_id = "theshoaibme/PanOrganNet"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"HuggingFace Downloader: {dataset_id}")
    print(f"Target Directory: {DATASET_DIR}")
    print(f"==================================================")

    # Check if dataset is already downloaded and extracted
    existing_files = [f for f in DATASET_DIR.glob("*") if not f.name.startswith(".")]
    if len(existing_files) > 1:
        print(f"[SKIP] {dataset_id} is already downloaded and present in {DATASET_DIR}\n")
        return

    try:
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(DATASET_DIR),
            token=token
        )
        print(f"SUCCESS: Downloaded HuggingFace dataset to: {path}\n")
    except Exception as e:
        print(f"ERROR downloading HuggingFace dataset: {e}\n")

if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else None
    download_hf_dataset(token_arg)
