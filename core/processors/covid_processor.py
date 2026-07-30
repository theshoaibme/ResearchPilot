import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "COVID19_XRay"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_processor import apply_clahe_2d, resize_2d

def process_covid():
    print("==================================================")
    print("Processing Dataset: COVID-19 Radiography Database")
    print("==================================================")

    candidates = [
        RAW_DIR / "COVID19_XRay" / "COVID-19_Radiography_Dataset",
        RAW_DIR / "COVID19_XRay",
        RAW_DIR / "PanOrganNet" / "COVID19_XRay"
    ]
    covid_raw = None
    for cand in candidates:
        if cand.exists() and any(cand.glob("**/*.png")):
            covid_raw = cand
            break

    if not covid_raw:
        print(f"COVID-19 Radiography raw images not found in: {candidates}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    categories = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
    processed_count = 0

    for cat in categories:
        cat_path = covid_raw / cat
        if not cat_path.exists():
            continue

        out_cat_path = PROCESSED_DIR / cat
        out_cat_path.mkdir(parents=True, exist_ok=True)

        images_dir = cat_path / "images" if (cat_path / "images").exists() else cat_path

        for img_file in list(images_dir.glob("*.png"))[:100]:  # Process available cohort subset
            img = np.array(Image.open(img_file).convert("L"))
            img_processed = apply_clahe_2d(img)
            img_resized = resize_2d(img_processed, target_size=(512, 512), is_mask=False)

            out_file = out_cat_path / f"{img_file.stem}.npz"
            np.savez_compressed(out_file, image=img_resized)
            processed_count += 1

    print(f"SUCCESS: Processed {processed_count} COVID-19 X-Ray samples into {PROCESSED_DIR}\n")

if __name__ == "__main__":
    process_covid()
