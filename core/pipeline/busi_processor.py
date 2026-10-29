import os
import shutil
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class BusiProcessor:
    """
    Standardizes the BUSI (Breast Ultrasound Images) dataset for the Oncology module.
    Classes: Normal (0), Benign (1), Malignant (2).
    Separates ultrasound scan images from binary ground-truth segmentation masks.
    """
    def __init__(self, base_path: str = "dataset"):
        self.raw_dir = Path(base_path) / "raw" / "BUSI" / "Dataset_BUSI_with_GT"
        self.processed_dir = Path(base_path) / "processed" / "oncology" / "busi"
        self.splits_dir = Path(base_path) / "splits" / "oncology" / "busi"

        self.dirs = {
            'images': self.processed_dir / "images",
            'masks': self.processed_dir / "masks",
            'labels': self.processed_dir / "labels",
            'manifests': self.processed_dir / "manifests",
            'statistics': self.processed_dir / "statistics"
        }

        self.label_map = {
            "normal": 0,
            "benign": 1,
            "malignant": 2
        }

    def _scaffold(self):
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}=== Processing BUSI Breast Ultrasound Dataset ==={Colors.RESET}")
        if not self.raw_dir.exists():
            print(f"{Colors.RED}[ERROR] BUSI raw directory not found: {self.raw_dir}{Colors.RESET}")
            return

        self._scaffold()
        categories = ["normal", "benign", "malignant"]

        all_labels = []
        stats = {}
        global_img_count = 0

        for cat in categories:
            cat_dir = self.raw_dir / cat
            if not cat_dir.exists():
                continue

            class_id = self.label_map[cat]
            all_files = sorted(os.listdir(cat_dir))

            # Filter images: exclude mask files
            img_files = [f for f in all_files if f.endswith('.png') and '_mask' not in f]
            stats[cat] = len(img_files)
            print(f"{Colors.CYAN}Processing {cat}: {len(img_files)} ultrasound scans...{Colors.RESET}")

            pbar = tqdm(img_files, desc=cat, leave=False)
            for img_name in pbar:
                global_img_count += 1
                new_img_name = f"{cat}_{global_img_count:05d}.png"

                # Copy image
                src_img = cat_dir / img_name
                dst_img = self.dirs['images'] / new_img_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

                # Copy corresponding mask(s) if present
                base_stem = img_name.replace('.png', '')
                mask_files = [f for f in all_files if f.startswith(base_stem) and '_mask' in f]
                for m_idx, mask_file in enumerate(mask_files):
                    mask_suffix = f"_{m_idx}" if m_idx > 0 else ""
                    new_mask_name = f"{cat}_{global_img_count:05d}_mask{mask_suffix}.png"
                    shutil.copy2(cat_dir / mask_file, self.dirs['masks'] / new_mask_name)

                all_labels.append({
                    'image_id': global_img_count,
                    'filename': new_img_name,
                    'original_filename': img_name,
                    'class': cat,
                    'class_id': class_id,
                    'has_mask': len(mask_files) > 0
                })

        # Save labels CSV
        labels_df = pd.DataFrame(all_labels)
        labels_csv = self.dirs['labels'] / "labels.csv"
        labels_df.to_csv(labels_csv, index=False)
        print(f"{Colors.GREEN}[SUCCESS] Saved {len(labels_df)} samples to {labels_csv}{Colors.RESET}")

        # Stratified Split (70/15/15)
        train_df, temp_df = train_test_split(labels_df, test_size=0.30, stratify=labels_df['class'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['class'], random_state=42)

        train_df.to_csv(self.splits_dir / "train.csv", index=False)
        val_df.to_csv(self.splits_dir / "validation.csv", index=False)
        test_df.to_csv(self.splits_dir / "test.csv", index=False)
        print(f"{Colors.GREEN}[SUCCESS] Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}{Colors.RESET}")

        # Save Statistics
        stats_data = {
            "dataset_name": "BUSI Breast Ultrasound",
            "total_scans": len(labels_df),
            "class_distribution": stats,
            "split_distribution": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            }
        }
        with open(self.dirs['statistics'] / "dataset_statistics.json", "w") as f:
            json.dump(stats_data, f, indent=4)

        print(f"{Colors.BOLD}{Colors.GREEN}🎉 BUSI Dataset Processing Completed Successfully!{Colors.RESET}\n")

if __name__ == "__main__":
    processor = BusiProcessor()
    processor.process()
