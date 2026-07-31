import os
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw" / "TotalSegmentator"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "TotalSegmentator"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_processor import ct_hu_windowing, resample_volume_3d

def process_totalsegmentator():
    print("==================================================")
    print("Processing Dataset: TotalSegmentator (v2 3D CT)")
    print("==================================================")

    if not RAW_DIR.exists():
        print(f"TotalSegmentator raw directory not found: {RAW_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    try:
        import nibabel as nib
    except ImportError:
        os.system(f"{sys.executable} -m pip install nibabel")
        import nibabel as nib

    nii_files = list(RAW_DIR.glob("**/*.nii.gz"))
    if not nii_files:
        print(f"No .nii.gz files found in {RAW_DIR} yet (download may still be in progress).")
        return

    for nii_file in nii_files[:10]:  # Process available volumetric batch
        try:
            nii = nib.load(str(nii_file))
            volume = nii.get_fdata().astype(np.float32)
            current_spacing = nii.header.get_zooms()[:3]

            # 1. CT Soft Tissue HU Windowing [-150, 250]
            windowed = ct_hu_windowing(volume, hu_min=-150, hu_max=250)

            # 2. 3D Isotropic Spatial Resampling (1.5mm x 1.5mm x 1.5mm)
            resampled = resample_volume_3d(windowed, current_spacing=current_spacing, target_spacing=(1.5, 1.5, 1.5))

            out_file = PROCESSED_DIR / f"{nii_file.stem.replace('.nii', '')}.npz"
            np.savez_compressed(out_file, volume=resampled)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {nii_file.name}: {e}")

    print(f"SUCCESS: Processed {processed_count} TotalSegmentator 3D CT volumes into {PROCESSED_DIR}\n")

if __name__ == "__main__":
    process_totalsegmentator()
