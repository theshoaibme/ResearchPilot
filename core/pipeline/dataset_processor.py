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

class DatasetProcessor:
    def __init__(self, module_name: str, dataset_name: str, base_path: str = "dataset"):
        self.module_name = module_name
        self.dataset_name = dataset_name
        
        # Paths
        self.raw_dir = Path(base_path) / "raw" / module_name / dataset_name
        self.processed_dir = Path(base_path) / "processed" / module_name / dataset_name
        self.splits_dir = Path(base_path) / "splits" / module_name / dataset_name
        
        # Processed Subdirectories
        self.dirs = {
            'images': self.processed_dir / "images",
            'labels': self.processed_dir / "labels",
            'metadata': self.processed_dir / "metadata",
            'manifests': self.processed_dir / "manifests",
            'statistics': self.processed_dir / "statistics",
            'reports': self.processed_dir / "reports",
            'cache': self.processed_dir / "cache",
            'configs': self.processed_dir / "configs"
        }
        
        # Standardized Label Map (IMMUTABLE)
        self.label_map = {
            "Normal": 0,
            "COVID": 1,
            "Lung_Opacity": 2,
            "Viral_Pneumonia": 3
        }

    def _scaffold_directories(self):
        print(f"{Colors.CYAN}Scaffolding directories for {self.dataset_name}...{Colors.RESET}")
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self):
        print(f"{Colors.BOLD}{Colors.MAGENTA}=== Starting Production Data Pipeline ==={Colors.RESET}")
        
        if not self.raw_dir.exists():
            print(f"{Colors.RED}[ERROR] Raw directory not found: {self.raw_dir}{Colors.RESET}")
            return
            
        self._scaffold_directories()
        
        # We will parse raw folders: 'Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia'
        raw_folders = [f for f in os.listdir(self.raw_dir) if os.path.isdir(self.raw_dir / f)]
        
        all_labels = []
        all_metadata = []
        stats = {}
        
        global_img_count = 0
        
        for folder in raw_folders:
            # Map original names to standard names
            std_class = folder.replace(" ", "_")
            if std_class not in self.label_map:
                print(f"{Colors.YELLOW}[WARN] Ignoring unmapped folder: {folder}{Colors.RESET}")
                continue
                
            class_id = self.label_map[std_class]
            img_dir = self.raw_dir / folder / "images"
            
            if not img_dir.exists():
                print(f"{Colors.YELLOW}[WARN] No images folder in {folder}{Colors.RESET}")
                continue
                
            images = [img for img in os.listdir(img_dir) if img.endswith('.png')]
            stats[std_class] = len(images)
            
            # Read metadata if exists
            meta_file = self.raw_dir / f"{folder}.metadata.xlsx"
            df_meta = None
            if meta_file.exists():
                try:
                    df_meta = pd.read_excel(meta_file)
                except Exception as e:
                    print(f"{Colors.RED}[ERROR] Could not read {meta_file}: {e}{Colors.RESET}")
            
            print(f"{Colors.CYAN}Processing {std_class} ({len(images)} images)...{Colors.RESET}")
            
            # Copy images and build label/metadata structures
            pbar = tqdm(images, desc=std_class, leave=False)
            for img_name in pbar:
                global_img_count += 1
                
                # Create standard filename: class_000001.png
                new_filename = f"{std_class.lower()}_{global_img_count:06d}.png"
                
                # File copy
                src_path = img_dir / img_name
                dst_path = self.dirs['images'] / new_filename
                
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    
                # Store label info
                all_labels.append({
                    'image_id': global_img_count,
                    'filename': new_filename,
                    'class': std_class,
                    'class_id': class_id
                })
                
                # Match metadata (Very naive match since patient_id doesn't exist, we just map by FILE NAME)
                if df_meta is not None:
                    # original filenames often have extensions missing in the excel sheet 'FILE NAME'
                    # e.g., 'Normal-1184' in excel but 'Normal-1184.png' in folder
                    base_name = img_name.replace('.png', '')
                    match = df_meta[df_meta['FILE NAME'] == base_name]
                    if not match.empty:
                        row = match.iloc[0].to_dict()
                        row['image_id'] = global_img_count
                        row['new_filename'] = new_filename
                        all_metadata.append(row)
        
        # Save Labels CSV
        labels_df = pd.DataFrame(all_labels)
        labels_csv_path = self.dirs['labels'] / "labels.csv"
        labels_df.to_csv(labels_csv_path, index=False)
        print(f"{Colors.GREEN}[SUCCESS] Saved {len(labels_df)} labels to {labels_csv_path}{Colors.RESET}")
        
        # Save Metadata CSV/Parquet
        if all_metadata:
            meta_df = pd.DataFrame(all_metadata)
            meta_csv_path = self.dirs['metadata'] / "patients.csv"
            meta_df.to_csv(meta_csv_path, index=False)
            try:
                meta_df.to_parquet(self.dirs['metadata'] / "patients.parquet")
            except ImportError:
                print(f"{Colors.YELLOW}[WARN] pyarrow/fastparquet not installed. Skipping .parquet{Colors.RESET}")
            print(f"{Colors.GREEN}[SUCCESS] Saved metadata to {meta_csv_path}{Colors.RESET}")
            
        # Stratified Split (70/15/15)
        # Since patient ID is missing, we perform image-level stratified split on the class column.
        print(f"{Colors.CYAN}Performing Image-Level Stratified Split...{Colors.RESET}")
        train_df, temp_df = train_test_split(labels_df, test_size=0.30, stratify=labels_df['class'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['class'], random_state=42)
        
        train_df.to_csv(self.splits_dir / "train.csv", index=False)
        val_df.to_csv(self.splits_dir / "validation.csv", index=False)
        test_df.to_csv(self.splits_dir / "test.csv", index=False)
        
        print(f"{Colors.GREEN}[SUCCESS] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}{Colors.RESET}")
        
        # Statistics
        stats_data = {
            "dataset_name": self.dataset_name,
            "total_images": len(labels_df),
            "class_distribution": stats,
            "split_distribution": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            }
        }
        
        with open(self.dirs['statistics'] / "dataset_statistics.json", "w") as f:
            json.dump(stats_data, f, indent=4)
            
        pd.DataFrame(list(stats.items()), columns=['Class', 'Images']).to_csv(self.dirs['statistics'] / "class_distribution.csv", index=False)
        
        # Manifest
        manifest = {
            "name": f"{self.dataset_name} Database",
            "version": "1.0",
            "images": len(labels_df),
            "classes": len(self.label_map),
            "format": "png",
            "labels_file": str(labels_csv_path.absolute()),
            "splits": {
                "train": str((self.splits_dir / "train.csv").absolute()),
                "validation": str((self.splits_dir / "validation.csv").absolute()),
                "test": str((self.splits_dir / "test.csv").absolute())
            }
        }
        with open(self.dirs['manifests'] / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
            
        print(f"{Colors.BOLD}{Colors.GREEN}=== Dataset Processing Complete! ==={Colors.RESET}")

if __name__ == "__main__":
    processor = DatasetProcessor(module_name="pulmonology", dataset_name="chest_xray")
    processor.process()
