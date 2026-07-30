import os
import sys
import zipfile
from pathlib import Path

# Configure Kaggle Token
KAGGLE_TOKEN = "KGAT_7779efdbcabb625d5aaffdb2a39465c4"
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting", unit="file"):
                zip_ref.extract(member, extract_to)
        print(f"Extracted {zip_path.name} successfully.")
        zip_path.unlink()
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile, Exception) as e:
        print(f"Zip extraction failed for {zip_path.name}: {e}")
        print(f"Removing corrupted archive: {zip_path}")
        if zip_path.exists():
            zip_path.unlink()
        return False

def download_kaggle_dataset(api, dataset_id, target_folder, is_competition=False):
    target_path = DATASET_BASE / target_folder
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"Kaggle Downloader: {dataset_id}")
    print(f"Destination: {target_path}")
    print(f"==================================================")

    # 1. Skip if dataset is already extracted and present
    extracted_items = [f for f in target_path.glob("*") if not f.name.endswith(".zip") and not f.name.endswith(".kaggle-partial")]
    if len(extracted_items) > 0:
        print(f"[SKIP] Dataset '{dataset_id}' is already downloaded and extracted in {target_path}\n")
        return

    # 2. Clean lingering zip or partial files before fresh download
    for partial_file in target_path.glob("*.zip*"):
        try:
            partial_file.unlink()
        except Exception:
            pass

    # 3. Attempt download
    try:
        if is_competition:
            api.competition_download_files(dataset_id, path=str(target_path), force=True, quiet=False)
        else:
            api.dataset_download_files(dataset_id, path=str(target_path), unzip=False, force=True, quiet=False)

        # Unzip downloaded archives
        zip_files = list(target_path.glob("*.zip"))
        for zip_file in zip_files:
            unzip_with_progress(zip_file, target_path)

        print(f"SUCCESS: Completed {dataset_id}\n")
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "Forbidden" in err_msg:
            print(f"REQUIRES AGREEMENT: Please accept competition rules at https://www.kaggle.com/c/{dataset_id}/rules")
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
