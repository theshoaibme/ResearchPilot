import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_BASE = BASE_DIR / "dataset"

def download_tcia_collections():
    print("==================================================")
    print("TCIA Downloader (The Cancer Imaging Archive)")
    print("==================================================")

    try:
        from tcia_utils import nbia
    except ImportError:
        os.system(f"{sys.executable} -m pip install tcia-utils --break-system-packages")
        from tcia_utils import nbia

    collections = [
        {"name": "TCGA-KIRC", "dir": "TCGA_KIRC", "desc": "Renal Clear Cell Carcinoma CT/MRI"},
        {"name": "TCGA-LGG", "dir": "TCGA_LGG", "desc": "Brain Lower Grade Glioma MRI"},
        {"name": "PROSTATEx", "dir": "PROSTATEx", "desc": "Prostate Multi-parametric MRI"},
        {"name": "CBIS-DDSM", "dir": "CBIS_DDSM", "desc": "Breast Mammography DDSM"}
    ]

    for item in collections:
        target_path = DATASET_BASE / item["dir"]
        target_path.mkdir(parents=True, exist_ok=True)

        print(f"\nFetching series metadata for TCIA Collection: {item['name']} ({item['desc']})...")
        print(f"Target: {target_path}")

        try:
            series_data = nbia.getSeries(collection=item["name"])
            if series_data is not None and not series_data.empty:
                print(f"Found {len(series_data)} series. Starting download...")
                nbia.downloadSeries(series_data, path=str(target_path))
                print(f"SUCCESS: Completed TCIA Collection {item['name']}\n")
            else:
                print(f"No series found for collection {item['name']}\n")
        except Exception as e:
            print(f"ERROR downloading TCIA Collection {item['name']}: {e}\n")

if __name__ == "__main__":
    download_tcia_collections()
