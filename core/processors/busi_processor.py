import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "BUSI"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_processor import apply_clahe_2d, resize_2d, normalize_min_max

def process_busi():
    print("==================================================")
    print("Processing Dataset: BUSI (Breast Ultrasound)")
    print("==================================================")

    # Locate BUSI raw folder
    busi_candidates = [
        RAW_DIR / "BUSI" / "Dataset_BUSI_with_GT",
        RAW_DIR / "PanOrganNet" / "BUSI" / "Dataset_BUSI_with_GT"
    ]
    busi_raw = None
    for cand in busi_candidates:
        if cand.exists():
            busi_raw = cand
            break

    if not busi_raw:
        print(f"BUSI raw data not found in candidates: {busi_candidates}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    categories = ["normal", "benign", "malignant"]
    processed_count = 0

    for cat in categories:
        cat_path = busi_raw / cat
        if not cat_path.exists():
            continue

        out_cat_path = PROCESSED_DIR / cat
        out_cat_path.mkdir(parents=True, exist_ok=True)

        for img_file in cat_path.glob("*.png"):
            if "_mask" in img_file.name:
                continue

            # Load image
            img = np.array(Image.open(img_file).convert("L"))
            img_processed = apply_clahe_2d(img)
            img_resized = resize_2d(img_processed, target_size=(512, 512), is_mask=False)

            # Look for mask file
            mask_name = img_file.stem + "_mask.png"
            mask_file = cat_path / mask_name
            mask_resized = None
            if mask_file.exists():
                mask_img = np.array(Image.open(mask_file).convert("L"))
                mask_bin = (mask_img > 128).astype(np.uint8)
                mask_resized = resize_2d(mask_bin, target_size=(512, 512), is_mask=True)

            # Save processed numpy payload
            out_file = out_cat_path / f"{img_file.stem}.npz"
            if mask_resized is not None:
                np.savez_compressed(out_file, image=img_resized, mask=mask_resized)
            else:
                np.savez_compressed(out_file, image=img_resized)

            processed_count += 1

    print(f"SUCCESS: Processed {processed_count} BUSI samples into {PROCESSED_DIR}\n")

if __name__ == "__main__":
    process_busi()
