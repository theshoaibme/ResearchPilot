import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
RAW_DDSM_DIR = PROJECT_ROOT / "dataset" / "raw" / "CBIS_DDSM"

def download_cbis_ddsm(batch_size: int = 20, max_series: int = 50):
    print("=" * 60)
    print("🎗️ CBIS-DDSM Breast Mammography Dataset: Download Pipeline")
    print("=" * 60)

    from tcia_utils import nbia

    RAW_DDSM_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Querying TCIA for collection 'CBIS-DDSM'...")
    series_list = nbia.getSeries(collection="CBIS-DDSM")
    if not series_list:
        print("[ERROR] Could not fetch series list from TCIA.")
        return

    df = pd.DataFrame(series_list)
    print(f"✓ Found {len(df)} total mammography series across {df['PatientID'].nunique()} patients.")

    if max_series is not None:
        df = df.head(max_series)
        print(f"Targeting first {max_series} mammography series.")

    # Filter out already downloaded series
    existing_uids = {p.name for p in RAW_DDSM_DIR.iterdir() if p.is_dir()}
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
            nbia.downloadSeries(chunk, path=str(RAW_DDSM_DIR), max_workers=6)
        except Exception as e:
            print(f"[Warning] Batch {batch_num} error: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 CBIS-DDSM Download Completed at: {RAW_DDSM_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    download_cbis_ddsm(batch_size=20, max_series=limit)
