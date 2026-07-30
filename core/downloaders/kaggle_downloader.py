import os
import sys
import zipfile
from pathlib import Path

# Configure Kaggle Token
KAGGLE_TOKEN = "KGAT_7779efdbcabb625d5aaffdb2a39465c4"
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_BASE = BASE_DIR / "dataset"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install kaggle tqdm --break-system-packages")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from tqdm import tqdm

def get_authenticated_api():
    api = KaggleApi()
    api.authenticate()
    return api

def unzip_with_progress(zip_path, extract_to):
    print(f"Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, extract_to)
    print(f"Extracted {zip_path.name} successfully.")

def download_kaggle_dataset(api, dataset_id, target_folder, is_competition=False):
    target_path = DATASET_BASE / target_folder
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"Kaggle Downloader: {dataset_id}")
    print(f"Destination: {target_path}")
    print(f"==================================================")

    try:
        if is_competition:
            api.competition_download_files(dataset_id, path=str(target_path), quiet=False)
        else:
            api.dataset_download_files(dataset_id, path=str(target_path), unzip=False, quiet=False)

        for zip_file in target_path.glob("*.zip"):
            unzip_with_progress(zip_file, target_path)
            zip_file.unlink()

        print(f"SUCCESS: Completed {dataset_id}\n")
    except Exception as e:
        print(f"ERROR downloading {dataset_id}: {e}\n")

def download_all_kaggle_datasets():
    api = get_authenticated_api()
    datasets = [
        {"id": "tawsifurrahman/covid19-radiography-database", "dir": "COVID19_XRay", "competition": False},
        {"id": "aryashah2k/breast-ultrasound-images-dataset", "dir": "BUSI", "competition": False},
        {"id": "ultrasound-nerve-segmentation", "dir": "Ultrasound_Nerve", "competition": True},
        {"id": "rsna-str-pulmonary-embolism-detection", "dir": "RSNA_PE", "competition": True},
        {"id": "prostate-cancer-grade-assessment", "dir": "PANDA_WSI", "competition": True},
    ]
    for ds in datasets:
        download_kaggle_dataset(api, ds["id"], ds["dir"], ds["competition"])

if __name__ == "__main__":
    download_all_kaggle_datasets()
