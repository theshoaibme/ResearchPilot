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

def normalize_dicom_slice(dcm) -> np.ndarray:
    pixels = dcm.pixel_array.astype(np.float32)
    modality = getattr(dcm, "Modality", "MR").upper()

    if modality == "CT":
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        hu = pixels * slope + intercept
        # Kidney / Soft tissue window: WL=40, WW=400 -> [-160, 240]
        hu_min = -160.0
        hu_max = 240.0
        clipped = np.clip(hu, hu_min, hu_max)
        norm = (clipped - hu_min) / (hu_max - hu_min)
    else:
        # MR: Percentile normalization
        p1, p99 = np.percentile(pixels, (1, 99))
        if p99 > p1:
            clipped = np.clip(pixels, p1, p99)
            norm = (clipped - p1) / (p99 - p1)
        else:
            norm = np.zeros_like(pixels)

    uint8_img = (norm * 255.0).astype(np.uint8)
    return uint8_img

def process_tcga_kirc(base_path: str = "dataset"):
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== Processing TCGA-KIRC (Renal CT & MRI) ==={Colors.RESET}")
    project_root = Path(base_path).resolve().parent
    raw_dir = Path(base_path) / "raw" / "TCGA_KIRC"
    meta_path = raw_dir / "metadata.csv"
    processed_dir = Path(base_path) / "processed" / "nephrology" / "tcga_kirc"
    splits_dir = Path(base_path) / "splits" / "nephrology" / "tcga_kirc"

    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"
    stats_dir = processed_dir / "statistics"
    reports_dir = processed_dir / "reports"

    for d in [images_dir, labels_dir, stats_dir, reports_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    print(f"{Colors.CYAN}Total indexed DICOM slices: {len(df_meta)} across {df_meta['patient_id'].nunique()} patients{Colors.RESET}")

    label_map = {"CT": 0, "MR": 1}
    processed_records = []

    print(f"{Colors.CYAN}Extracting and standardizing axial slices to PNG...{Colors.RESET}")
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Converting DICOMs"):
        rel_dcm_path = row["file_path"]
        full_dcm_path = raw_dir / rel_dcm_path

        if not full_dcm_path.exists():
            continue

        try:
            dcm = pydicom.dcmread(str(full_dcm_path))
            norm_array = normalize_dicom_slice(dcm)
            img = Image.fromarray(norm_array)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.BILINEAR)

            patient_id = str(row["patient_id"])
            modality = str(row["modality"]).upper()
            class_id = label_map.get(modality, 1)

            out_filename = f"{patient_id}_{modality}_{idx:05d}.png"
            out_img_path = images_dir / out_filename
            img.save(out_img_path)

            processed_records.append({
                "image_filename": out_filename,
                "relative_path": f"processed/nephrology/tcga_kirc/images/{out_filename}",
                "patient_id": patient_id,
                "modality": modality,
                "class_name": modality,
                "class_id": class_id,
                "series_description": str(row.get("series_description", "Unknown")),
                "width": 256,
                "height": 256
            })
        except Exception as e:
            continue

    df_processed = pd.DataFrame(processed_records)
    print(f"{Colors.GREEN}Successfully standardized {len(df_processed)} slices.{Colors.RESET}")

    # Patient-Aware Splitting (Zero patient leakage)
    print(f"{Colors.CYAN}Generating Patient-Aware (Subject-Level) 70/15/15 splits...{Colors.RESET}")
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

    # Save Label Map
    with open(labels_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    # Save Statistics
    stats = {
        "dataset_name": "TCGA-KIRC (Renal CT & MRI)",
        "total_images": len(df_processed),
        "total_patients": int(df_processed['patient_id'].nunique()),
        "classes": label_map,
        "class_distribution": df_processed['class_name'].value_counts().to_dict(),
        "splits": {
            "train": {"count": len(train_df), "patients": int(train_df['patient_id'].nunique())},
            "validation": {"count": len(val_df), "patients": int(val_df['patient_id'].nunique())},
            "test": {"count": len(test_df), "patients": int(test_df['patient_id'].nunique())}
        }
    }
    with open(stats_dir / "summary_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"{Colors.BOLD}{Colors.GREEN}✓ TCGA-KIRC Processing Completed!{Colors.RESET}")
    print(f"Train: {len(train_df)} slices ({train_df['patient_id'].nunique()} pts) | Val: {len(val_df)} slices ({val_df['patient_id'].nunique()} pts) | Test: {len(test_df)} slices ({test_df['patient_id'].nunique()} pts)")

if __name__ == "__main__":
    process_tcga_kirc()
