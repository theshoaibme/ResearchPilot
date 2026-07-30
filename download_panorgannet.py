import os
import sys
from huggingface_hub import snapshot_download

HF_TOKEN = "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"

def download_dataset(token=None):
    if not token:
        token = os.environ.get("HF_TOKEN", HF_TOKEN)
    
    if not token:
        print("Error: No Hugging Face token provided.")
        print("Please set HF_TOKEN environment variable or pass your token to download_dataset(token='...')")
        sys.exit(1)

    dataset_id = "the-shoaib2/PanOrganNet"
    output_dir = "dataset/PanOrganNet"

    print(f"Downloading {dataset_id} to {output_dir}...")
    try:
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=output_dir,
            token=token
        )
        print(f"Successfully downloaded dataset to: {path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else None
    download_dataset(token_arg)
