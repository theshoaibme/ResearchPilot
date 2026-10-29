import os
import shutil
import json
from pathlib import Path
import pandas as pd
from PIL import Image
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

class UltrasoundNerveProcessor:
    """
    Standardizes the Kaggle Ultrasound Nerve Segmentation Dataset for the Neurology track.
    Key Engineering:
      - Subject-Level (Patient-Wise) Stratification: Prevents patient-data leakage.
      - Image & Mask pairing with presence-of-nerve classification labels (binary: 0=no nerve, 1=nerve present).
      - Converts TIF to clean standard PNG for high-throughput PyTorch training.
    """
    def __init__(self, base_path: str = "dataset"):
        self.raw_dir = Path(base_path) / "raw" / "Ultrasound_Nerve" / "train"
        self.processed_dir = Path(base_path) / "processed" / "neurology" / "ultrasound_nerve"
        self.splits_dir = Path(base_path) / "splits" / "neurology" / "ultrasound_nerve"

        self.dirs = {
            'images': self.processed_dir / "images",
            'masks': self.processed_dir / "masks",
            'labels': self.processed_dir / "labels",
            'manifests': self.processed_dir / "manifests",
            'statistics': self.processed_dir / "statistics"
        }

    def _scaffold(self):
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== Processing Ultrasound Nerve Dataset ==={Colors.RESET}")
        if not self.raw_dir.exists():
            print(f"{Colors.RED}[ERROR] Raw nerve train directory not found: {self.raw_dir}{Colors.RESET}")
            return

        self._scaffold()
        all_files = sorted(os.listdir(self.raw_dir))

        # Filter ultrasound scans (exclude mask files)
        image_files = [f for f in all_files if f.endswith('.tif') and '_mask' not in f]
        print(f"{Colors.CYAN}Found {len(image_files)} ultrasound scans in {self.raw_dir}...{Colors.RESET}")

        all_records = []
        global_id = 0

        pbar = tqdm(image_files, desc="Standardizing Nerve Ultrasound", dynamic_ncols=True)
        for img_file in pbar:
            global_id += 1
            base_stem = img_file.replace('.tif', '')
            mask_file = f"{base_stem}_mask.tif"

            # Parse subject and scan number (e.g. 10_100 -> subject 10)
            parts = base_stem.split('_')
            subject_id = int(parts[0]) if parts[0].isdigit() else parts[0]
            scan_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else global_id

            # Destination PNG filenames
            new_img_name = f"nerve_{subject_id}_{scan_id:04d}.png"
            new_mask_name = f"nerve_{subject_id}_{scan_id:04d}_mask.png"

            src_img = self.raw_dir / img_file
            src_mask = self.raw_dir / mask_file

            dst_img = self.dirs['images'] / new_img_name
            dst_mask = self.dirs['masks'] / new_mask_name

            # Convert scan to PNG
            if not dst_img.exists():
                with Image.open(src_img) as im:
                    im.convert('L').save(dst_img)

            # Check if mask has any nerve pixels
            has_nerve = 0
            if src_mask.exists() and not dst_mask.exists():
                with Image.open(src_mask) as mim:
                    mim_l = mim.convert('L')
                    # Check if any white pixels exist (255)
                    extrema = mim_l.getextrema()
                    has_nerve = 1 if extrema[1] > 0 else 0
                    mim_l.save(dst_mask)
            elif dst_mask.exists():
                with Image.open(dst_mask) as mim_l:
                    extrema = mim_l.getextrema()
                    has_nerve = 1 if extrema[1] > 0 else 0

            all_records.append({
                'image_id': global_id,
                'filename': new_img_name,
                'mask_filename': new_mask_name,
                'subject_id': subject_id,
                'class': 'nerve_present' if has_nerve == 1 else 'no_nerve',
                'class_id': has_nerve
            })

        # Save labels CSV
        df = pd.DataFrame(all_records)
        labels_csv = self.dirs['labels'] / "labels.csv"
        df.to_csv(labels_csv, index=False)
        print(f"\n{Colors.GREEN}[SUCCESS] Saved {len(df)} records to {labels_csv}{Colors.RESET}")

        # Patient-Level (Subject-Aware) Group Split (70/15/15)
        gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss_train.split(df, groups=df['subject_id']))
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['subject_id']))
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]

        train_df.to_csv(self.splits_dir / "train.csv", index=False)
        val_df.to_csv(self.splits_dir / "validation.csv", index=False)
        test_df.to_csv(self.splits_dir / "test.csv", index=False)

        print(f"{Colors.GREEN}[SUCCESS] Patient-Aware Splits Created:{Colors.RESET}")
        print(f"  Train: {len(train_df)} samples ({train_df['subject_id'].nunique()} subjects)")
        print(f"  Validation: {len(val_df)} samples ({val_df['subject_id'].nunique()} subjects)")
        print(f"  Test: {len(test_df)} samples ({test_df['subject_id'].nunique()} subjects)")

        # Statistics
        stats = {
            "dataset_name": "Ultrasound Nerve Segmentation",
            "total_scans": len(df),
            "total_subjects": int(df['subject_id'].nunique()),
            "class_distribution": df['class'].value_counts().to_dict(),
            "split_distribution": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            }
        }
        with open(self.dirs['statistics'] / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=4)

        print(f"{Colors.BOLD}{Colors.GREEN}🎉 Ultrasound Nerve Dataset Processed Successfully!{Colors.RESET}\n")

if __name__ == "__main__":
    processor = UltrasoundNerveProcessor()
    processor.process()
