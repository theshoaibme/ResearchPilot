import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def normalize_mg_slice(dcm) -> np.ndarray:
    pixels = dcm.pixel_array.astype(np.float32)
    p1, p99 = np.percentile(pixels, (1, 99))
    if p99 > p1:
        clipped = np.clip(pixels, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
    else:
        norm = np.zeros_like(pixels)

    uint8_img = (norm * 255.0).astype(np.uint8)
    return uint8_img

def process_cbis_ddsm(base_path: str = "dataset"):
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== Processing CBIS-DDSM (Mammography Lesions & ROI Masks) ==={Colors.RESET}")
    raw_dir = Path(base_path) / "raw" / "CBIS_DDSM"
    meta_path = raw_dir / "metadata.csv"
    processed_dir = Path(base_path) / "processed" / "oncology" / "cbis_ddsm"
    splits_dir = Path(base_path) / "splits" / "oncology" / "cbis_ddsm"

    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"
    labels_dir = processed_dir / "labels"
    stats_dir = processed_dir / "statistics"

    for d in [images_dir, masks_dir, labels_dir, stats_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    print(f"{Colors.CYAN}Total indexed Mammogram files: {len(df_meta)} across {df_meta['patient_id'].nunique()} cases{Colors.RESET}")

    processed_records = []

    print(f"{Colors.CYAN}Standardizing Mammography lesion images and ROI masks...{Colors.RESET}")
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Converting CBIS-DDSM"):
        rel_path = row["file_path"]
        full_path = raw_dir / rel_path

        if not full_path.exists():
            continue

        try:
            dcm = pydicom.dcmread(str(full_path))
            norm_array = normalize_mg_slice(dcm)
            img = Image.fromarray(norm_array)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.BILINEAR)

            patient_id = str(row["patient_id"])
            series_desc = str(row["series_description"]).lower()

            is_mask = 1 if "mask" in series_desc else 0
            is_full = 1 if "full" in series_desc else 0
            
            # Category: 0 = Lesion Crop, 1 = ROI Mask, 2 = Full Mammogram
            cat_id = 1 if is_mask else (2 if is_full else 0)
            cat_name = "ROI_Mask" if is_mask else ("Full_Mammogram" if is_full else "Lesion_Crop")

            target_folder = masks_dir if is_mask else images_dir
            out_filename = f"{patient_id}_{cat_name}_{idx:03d}.png"
            img.save(target_folder / out_filename)

            processed_records.append({
                "filename": out_filename,
                "relative_path": f"processed/oncology/cbis_ddsm/{'masks' if is_mask else 'images'}/{out_filename}",
                "patient_id": patient_id,
                "class_name": cat_name,
                "class_id": cat_id,
                "series_description": str(row["series_description"]),
                "width": 256,
                "height": 256
            })
        except Exception:
            continue

    df_processed = pd.DataFrame(processed_records)
    print(f"{Colors.GREEN}Successfully standardized {len(df_processed)} files.{Colors.RESET}")

    # Patient-Aware Splitting
    print(f"{Colors.CYAN}Generating Patient-Aware (Subject-Level) splits...{Colors.RESET}")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df_processed, groups=df_processed['patient_id']))

    train_df = df_processed.iloc[train_idx].copy()
    temp_df = df_processed.iloc[temp_idx].copy()

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['patient_id']))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "validation.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    label_map = {"Lesion_Crop": 0, "ROI_Mask": 1, "Full_Mammogram": 2}
    with open(labels_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    stats = {
        "dataset_name": "CBIS-DDSM Mammography",
        "total_files": len(df_processed),
        "total_patients": int(df_processed['patient_id'].nunique()),
        "classes": label_map,
        "splits": {
            "train": {"count": len(train_df), "patients": int(train_df['patient_id'].nunique())},
            "validation": {"count": len(val_df), "patients": int(val_df['patient_id'].nunique())},
            "test": {"count": len(test_df), "patients": int(test_df['patient_id'].nunique())}
        }
    }
    with open(stats_dir / "summary_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"{Colors.BOLD}{Colors.GREEN}✓ CBIS-DDSM Processing Completed!{Colors.RESET}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    process_cbis_ddsm()
