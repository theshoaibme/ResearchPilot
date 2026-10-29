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

def normalize_mr_slice(dcm) -> np.ndarray:
    pixels = dcm.pixel_array.astype(np.float32)
    p1, p99 = np.percentile(pixels, (1, 99))
    if p99 > p1:
        clipped = np.clip(pixels, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
    else:
        norm = np.zeros_like(pixels)

    uint8_img = (norm * 255.0).astype(np.uint8)
    return uint8_img

def process_prostatex(base_path: str = "dataset"):
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== Processing PROSTATEx (Multi-Parametric MRI) ==={Colors.RESET}")
    raw_dir = Path(base_path) / "raw" / "PROSTATEx"
    master_meta_path = Path(base_path) / "raw" / "master_metadata.csv"
    processed_dir = Path(base_path) / "processed" / "oncology" / "prostatex"
    splits_dir = Path(base_path) / "splits" / "oncology" / "prostatex"

    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"
    stats_dir = processed_dir / "statistics"
    reports_dir = processed_dir / "reports"

    for d in [images_dir, labels_dir, stats_dir, reports_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not master_meta_path.exists():
        raise FileNotFoundError(f"Master metadata not found: {master_meta_path}")

    df_master = pd.read_csv(master_meta_path)
    df_px = df_master[df_master['dataset_name'] == 'panorgan-prostatex-mri'].copy()
    print(f"{Colors.CYAN}Total indexed PROSTATEx slices: {len(df_px)} across {df_px['patient_id'].nunique()} patients{Colors.RESET}")

    # Binary sequence grouping / classification: T2 vs Diffusion/Other
    processed_records = []

    print(f"{Colors.CYAN}Extracting and standardizing axial slices to PNG...{Colors.RESET}")
    for idx, row in tqdm(df_px.iterrows(), total=len(df_px), desc="Converting PROSTATEx"):
        rel_dcm_path = row["file_path"]
        full_dcm_path = raw_dir / rel_dcm_path

        if not full_dcm_path.exists():
            continue

        try:
            dcm = pydicom.dcmread(str(full_dcm_path))
            norm_array = normalize_mr_slice(dcm)
            img = Image.fromarray(norm_array)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.BILINEAR)

            patient_id = str(row["patient_id"])
            series_desc = str(row.get("series_description", "")).lower()
            
            # Label: 0 = T2-Weighted Anatomical, 1 = Functional / Diffusion / ADC
            is_func = 1 if ("diff" in series_desc or "dwi" in series_desc or "adc" in series_desc or "ktrans" in series_desc) else 0
            class_name = "Functional_ADC_DWI" if is_func == 1 else "Anatomical_T2"

            out_filename = f"{patient_id}_{idx:05d}.png"
            out_img_path = images_dir / out_filename
            img.save(out_img_path)

            processed_records.append({
                "image_filename": out_filename,
                "relative_path": f"processed/oncology/prostatex/images/{out_filename}",
                "patient_id": patient_id,
                "class_name": class_name,
                "class_id": is_func,
                "series_description": str(row.get("series_description", "Unknown")),
                "width": 256,
                "height": 256
            })
        except Exception:
            continue

    df_processed = pd.DataFrame(processed_records)
    print(f"{Colors.GREEN}Successfully standardized {len(df_processed)} slices.{Colors.RESET}")

    # Patient-Aware Splitting
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

    label_map = {"Anatomical_T2": 0, "Functional_ADC_DWI": 1}
    with open(labels_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    stats = {
        "dataset_name": "PROSTATEx (Multi-Parametric Prostate MRI)",
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

    print(f"{Colors.BOLD}{Colors.GREEN}✓ PROSTATEx Processing Completed!{Colors.RESET}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    process_prostatex()
