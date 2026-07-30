import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_downloader import download_hf_dataset
from kaggle_downloader import download_all_kaggle_datasets
from tcia_downloader import download_tcia_collections
from open_access_downloader import download_totalsegmentator

def run_all_downloaders():
    print("==================================================")
    print("Pan-Organ Net: Master Dataset Downloader Suite")
    print("==================================================\n")

    print("[1/4] Running Hugging Face Downloader...")
    try:
        download_hf_dataset()
    except Exception as e:
        print(f"HF Downloader Error: {e}")

    print("\n[2/4] Running Kaggle Downloader...")
    try:
        download_all_kaggle_datasets()
    except Exception as e:
        print(f"Kaggle Downloader Error: {e}")

    print("\n[3/4] Running TCIA Downloader...")
    try:
        download_tcia_collections()
    except Exception as e:
        print(f"TCIA Downloader Error: {e}")

    print("\n[4/4] Running Open Access (TotalSegmentator) Downloader...")
    try:
        download_totalsegmentator()
    except Exception as e:
        print(f"Open Access Downloader Error: {e}")

    print("\n==================================================")
    print("All Automated Downloads Complete!")
    print("==================================================")

if __name__ == "__main__":
    run_all_downloaders()
