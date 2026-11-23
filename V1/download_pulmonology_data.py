import os
import sys
import shutil
import zipfile
from pathlib import Path

# Kaggle credentials
os.environ.setdefault("KAGGLE_USERNAME", "theshoaib2")
os.environ.setdefault("KAGGLE_KEY", "5cf6cabce5347ceeec86c8448f848d9f")

from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
RAW_PULMONOLOGY_DIR = PROJECT_ROOT / "dataset" / "raw" / "pulmonology" / "chest_xray"
RAW_COVID_DIR = PROJECT_ROOT / "dataset" / "raw" / "COVID19_XRay"
TEMP_DOWNLOAD_DIR = PROJECT_ROOT / "dataset" / "temp_download"

def download_and_setup_pulmonology():
    print("=" * 60)
    print("🫁 Pulmonology Dataset: Complete Download & Processing Pipeline")
    print("=" * 60)

    # 1. Authenticate
    print("\n[1/5] Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    print("✓ Successfully authenticated.")

    # 2. Setup directories
    TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PULMONOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_COVID_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = "tawsifurrahman/covid19-radiography-database"
    print(f"\n[2/5] Downloading dataset: {dataset_id}...")
    api.dataset_download_files(dataset_id, path=str(TEMP_DOWNLOAD_DIR), unzip=False, force=True, quiet=False)

    # 3. Extracting archive
    zip_files = list(TEMP_DOWNLOAD_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {TEMP_DOWNLOAD_DIR}")
    
    zip_path = zip_files[0]
    print(f"\n[3/5] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Unzipping archive", unit="file"):
            zip_ref.extract(member, TEMP_DOWNLOAD_DIR)
    
    # Remove zip to save disk space
    if zip_path.exists():
        zip_path.unlink()

    # 4. Organizing directory structure
    print("\n[4/5] Organizing raw dataset files...")
    # Locate extracted dataset directory
    extracted_candidates = [
        TEMP_DOWNLOAD_DIR / "COVID-19_Radiography_Dataset",
        TEMP_DOWNLOAD_DIR / "covid19-radiography-database",
        TEMP_DOWNLOAD_DIR
    ]
    extracted_root = None
    for cand in extracted_candidates:
        if cand.exists() and ((cand / "COVID").exists() or (cand / "Normal").exists()):
            extracted_root = cand
            break
            
    if not extracted_root:
        raise FileNotFoundError(f"Could not locate extracted category folders in {TEMP_DOWNLOAD_DIR}")

    print(f"Found source data at: {extracted_root}")

    # Copy / Move to RAW_PULMONOLOGY_DIR
    categories = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
    for item in extracted_root.iterdir():
        target_dest = RAW_PULMONOLOGY_DIR / item.name
        if target_dest.exists():
            if target_dest.is_dir():
                shutil.rmtree(target_dest)
            else:
                target_dest.unlink()
        shutil.move(str(item), str(target_dest))
        print(f"  ✓ Moved {item.name} -> {target_dest}")

    # Clean up temp folder
    shutil.rmtree(TEMP_DOWNLOAD_DIR, ignore_errors=True)

    # Also link or copy to RAW_COVID_DIR for backward compatibility
    for cat in RAW_PULMONOLOGY_DIR.iterdir():
        covid_dest = RAW_COVID_DIR / cat.name
        if not covid_dest.exists():
            try:
                os.symlink(str(cat.resolve()), str(covid_dest))
            except OSError:
                if cat.is_dir():
                    shutil.copytree(str(cat), str(covid_dest))
                else:
                    shutil.copy2(str(cat), str(covid_dest))

    print(f"✓ Raw Pulmonology data ready at: {RAW_PULMONOLOGY_DIR}")

    # 5. Process dataset using production DatasetProcessor
    print("\n[5/5] Executing Production DatasetProcessor...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.pipeline.dataset_processor import DatasetProcessor
    processor = DatasetProcessor(module_name="pulmonology", dataset_name="chest_xray")
    processor.process()

    print("\n" + "=" * 60)
    print("🎉 Pulmonology Dataset Download & Processing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    download_and_setup_pulmonology()
