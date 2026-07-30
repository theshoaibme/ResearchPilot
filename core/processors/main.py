import sys
from pathlib import Path

# Add core/processors to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from busi_processor import process_busi
from covid_processor import process_covid
from hf_processor import process_hf_dataset
from totalseg_processor import process_totalsegmentator
from nerve_processor import process_ultrasound_nerve

def run_all_processors():
    print("==================================================")
    print("Pan-Organ Net: Master Dataset Processing Suite")
    print("==================================================\n")

    print("[1/5] Processing BUSI Breast Ultrasound Dataset...")
    try:
        process_busi()
    except Exception as e:
        print(f"BUSI Processor Error: {e}\n")

    print("[2/5] Processing COVID-19 Radiography Dataset...")
    try:
        process_covid()
    except Exception as e:
        print(f"COVID Processor Error: {e}\n")

    print("[3/5] Processing PanOrganNet (Hugging Face) Dataset...")
    try:
        process_hf_dataset()
    except Exception as e:
        print(f"HF Processor Error: {e}\n")

    print("[4/5] Processing Ultrasound Nerve Segmentation Dataset...")
    try:
        process_ultrasound_nerve()
    except Exception as e:
        print(f"Nerve Processor Error: {e}\n")

    print("[5/5] Processing TotalSegmentator 3D CT Dataset...")
    try:
        process_totalsegmentator()
    except Exception as e:
        print(f"TotalSegmentator Processor Error: {e}\n")

    print("==================================================")
    print("All Data Processing Completed!")
    print("Output directory: dataset/processed/")
    print("==================================================")

if __name__ == "__main__":
    run_all_processors()
