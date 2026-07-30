import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw" / "Ultrasound_Nerve"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "Ultrasound_Nerve"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_processor import apply_clahe_2d, resize_2d

def process_ultrasound_nerve():
    print("==================================================")
    print("Processing Dataset: Ultrasound Nerve Segmentation")
    print("==================================================")

    if not RAW_DIR.exists():
        print(f"Ultrasound Nerve raw directory not found: {RAW_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = RAW_DIR / "train"
    if not train_dir.exists():
        print(f"Train directory not found in {RAW_DIR}")
        return

    images = [f for f in train_dir.glob("*.tif") if not f.name.endswith("_mask.tif")]
    processed_count = 0

    print(f"Found {len(images)} nerve ultrasound training images.")
    for img_path in images:
        mask_path = train_dir / f"{img_path.stem}_mask.tif"
        try:
            with Image.open(img_path) as img_obj:
                img_arr = np.array(img_obj.convert("L"), dtype=np.float32) / 255.0

            img_clahe = apply_clahe_2d(img_arr)
            img_resized = resize_2d(img_clahe, target_size=(512, 512), is_mask=False)

            mask_resized = None
            if mask_path.exists():
                with Image.open(mask_path) as mask_obj:
                    mask_arr = (np.array(mask_obj.convert("L")) > 127).astype(np.float32)
                mask_resized = resize_2d(mask_arr, target_size=(512, 512), is_mask=True)

            out_file = PROCESSED_DIR / f"{img_path.stem}.npz"
            if mask_resized is not None:
                np.savez_compressed(out_file, image=img_resized, mask=mask_resized)
            else:
                np.savez_compressed(out_file, image=img_resized)

            processed_count += 1
            if processed_count % 500 == 0 or processed_count == len(images):
                print(f"Processed {processed_count}/{len(images)} nerve ultrasound scans...")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print(f"SUCCESS: Saved {processed_count} preprocessed arrays to {PROCESSED_DIR}\n")

if __name__ == "__main__":
    process_ultrasound_nerve()
