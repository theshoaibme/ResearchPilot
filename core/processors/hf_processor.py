import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "dataset" / "PanOrganNet"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "PanOrganNet"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_processor import apply_clahe_2d, resize_2d

def process_hf_dataset():
    print("==================================================")
    print("Processing Dataset: PanOrganNet (Hugging Face)")
    print("==================================================")

    if not RAW_DIR.exists():
        print(f"PanOrganNet raw directory not found: {RAW_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    # Search for sample images or subfolders inside PanOrganNet
    for img_file in list(RAW_DIR.glob("**/*.png"))[:100]:
        if "_mask" in img_file.name:
            continue

        try:
            img = np.array(Image.open(img_file).convert("L"))
            img_processed = apply_clahe_2d(img)
            img_resized = resize_2d(img_processed, target_size=(512, 512), is_mask=False)

            rel_path = img_file.relative_to(RAW_DIR)
            out_file = PROCESSED_DIR / rel_path.parent / f"{img_file.stem}.npz"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(out_file, image=img_resized)
            processed_count += 1
        except Exception as e:
            continue

    print(f"SUCCESS: Processed {processed_count} PanOrganNet samples into {PROCESSED_DIR}\n")

if __name__ == "__main__":
    process_hf_dataset()
