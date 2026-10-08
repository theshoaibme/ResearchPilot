import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
RAW_KIRC_DIR = PROJECT_ROOT / "dataset" / "raw" / "TCGA_KIRC"

def download_tcga_kirc(batch_size: int = 20, max_patients: int = 15):
    print("=" * 60)
    print("🫘 TCGA-KIRC Renal CT/MRI Dataset: Download Pipeline")
    print("=" * 60)

    from tcia_utils import nbia

    RAW_KIRC_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Querying TCIA for collection 'TCGA-KIRC'...")
    series_list = nbia.getSeries(collection="TCGA-KIRC")
    if not series_list:
        print("[ERROR] Could not fetch series list from TCIA.")
        return

    df = pd.DataFrame(series_list)
    print(f"✓ Found {len(df)} total series across {df['PatientID'].nunique()} patients.")

    patients = df["PatientID"].unique().tolist()
    if max_patients is not None:
        patients = patients[:max_patients]
        df = df[df["PatientID"].isin(patients)]
        print(f"Targeting first {max_patients} patients ({len(df)} series).")

    # Filter out already downloaded series
    existing_uids = {p.name for p in RAW_KIRC_DIR.iterdir() if p.is_dir()}
    pending_df = df[~df["SeriesInstanceUID"].isin(existing_uids)]
    print(f"Already downloaded: {len(existing_uids)} | Pending download: {len(pending_df)}")

    if len(pending_df) == 0:
        print("✓ All targeted series are already downloaded!")
        return

    print(f"\n[2/3] Downloading {len(pending_df)} series in batches with parallel workers...")
    records = pending_df.to_dict(orient="records")
    
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(records) + batch_size - 1) // batch_size
        print(f"\nDownloading Batch [{batch_num}/{total_batches}] ({len(chunk)} series)...")
        try:
            nbia.downloadSeries(chunk, path=str(RAW_KIRC_DIR), max_workers=6)
        except Exception as e:
            print(f"[Warning] Batch {batch_num} error: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 TCGA-KIRC Download Completed at: {RAW_KIRC_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    download_tcga_kirc(batch_size=20, max_patients=limit)
