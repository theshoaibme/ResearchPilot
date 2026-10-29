import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def publish_kirc_dataset():
    api = HfApi()
    username = api.whoami(token=HF_TOKEN)["name"]
    repo_id = f"{username}/panorgan-tcga-kirc-renal-ct-mri"

    splits_dir = PROJECT_ROOT / "dataset" / "splits" / "nephrology" / "tcga_kirc"
    processed_dir = PROJECT_ROOT / "dataset" / "processed" / "nephrology" / "tcga_kirc"

    api.create_repo(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN, exist_ok=True, private=False)

    print(f"Uploading splits to {repo_id}...")
    api.upload_folder(
        folder_path=str(splits_dir),
        path_in_repo="splits",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add patient-aware 70/15/15 train, validation, and test splits (zero patient leakage)"
    )

    for folder in ["labels", "statistics", "reports"]:
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

    print(f"✓ TCGA-KIRC Dataset Hub Updated: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    publish_kirc_dataset()
