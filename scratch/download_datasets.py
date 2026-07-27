import os
import urllib.request
import sys

dataset_dir = r"c:\Users\HP\OneDrive\Desktop\ResearchPilot\datasets"
os.makedirs(dataset_dir, exist_ok=True)

# Direct open-access sample files/packages for key benchmark datasets
direct_downloads = [
    {
        "name": "BUSI_Breast_Ultrasound",
        "folder": os.path.join(dataset_dir, "BUSI"),
        "url": "https://raw.githubusercontent.com/aryashah2k/Breast-Ultrasound-Images-Dataset/main/Dataset_BUSI_with_GT/benign/benign%20(1).png",
        "filename": "sample_benign_ultrasound.png"
    },
    {
        "name": "TotalSegmentator_Sample",
        "folder": os.path.join(dataset_dir, "TotalSegmentator"),
        "url": "https://zenodo.org/records/10047292/files/totalsegmentator_sample.zip?download=1",
        "filename": "totalsegmentator_sample.zip"
    },
    {
        "name": "IXI_Brain_MRI_Sample",
        "folder": os.path.join(dataset_dir, "IXI_Brain_MRI"),
        "url": "https://brain-development.org/ixi-dataset/IXI-T1.tar",
        "filename": "IXI_sample.tar"
    }
]

print("Starting dataset sample download pipeline...")

for ds in direct_downloads:
    target_folder = ds["folder"]
    os.makedirs(target_folder, exist_ok=True)
    target_file = os.path.join(target_folder, ds["filename"])
    
    print(f"\n[Downloading] {ds['name']} from {ds['url']} -> {target_file}")
    try:
        urllib.request.urlretrieve(ds["url"], target_file)
        size_mb = os.path.getsize(target_file) / (1024 * 1024)
        print(f"[Success] {ds['name']} downloaded ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[Download Error] {ds['name']}: {e}")

print("\nDirect download pipeline finished!")
