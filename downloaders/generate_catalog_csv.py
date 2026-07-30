import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "phases" / "Pan_Organ_Medical_Datasets_Catalog.csv"

# Comprehensive 23-dataset catalog across Kaggle, PhysioNet, TCIA, Zenodo, Grand Challenge, Synapse, and Mendeley Data
ultimate_dataset_catalog = [
    # --- 3D CT DATASETS ---
    {
        "Dataset ID": "DS-CT-01",
        "Dataset Name": "TotalSegmentator (v2)",
        "Platform": "Zenodo",
        "Primary Modality": "3D CT",
        "Target Organ Systems": "Whole Body (Thoracic, Abdominal, Pelvic, Musculoskeletal)",
        "Anatomical Structures / Labels": "117 organs, bones, vessels, tissue ground-truth masks",
        "Sample Size / Scale": "1,204 high-res CT volumes",
        "Primary Disease Taxonomy": "Neoplastic, Traumatic, Vascular, Autoimmune",
        "Target Medical Specialist": "Radiologist, Surgical Oncologist, Pulmonologist",
        "Data Access / Download Link": "https://zenodo.org/records/10047292",
        "Licensing": "CC BY 4.0 (Open Access)",
        "Format": "NIfTI (.nii.gz)",
        "Pre-training / Validation Role": "Pan-organ spatial segmentation ground-truth benchmark"
    },
    {
        "Dataset ID": "DS-CT-02",
        "Dataset Name": "LUNA16 / LIDC-IDRI",
        "Platform": "Grand Challenge / TCIA",
        "Primary Modality": "3D Low-Dose CT",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "1,018 lung nodule annotations with malignancy agreement",
        "Sample Size / Scale": "888 CT scans",
        "Primary Disease Taxonomy": "Neoplastic, Infectious, Traumatic",
        "Target Medical Specialist": "Pulmonologist, Radiologist",
        "Data Access / Download Link": "https://luna16.grand-challenge.org/",
        "Licensing": "Public Benchmark License",
        "Format": "MHD / Raw / NIfTI",
        "Pre-training / Validation Role": "Pulmonary nodule detection & AUC-ROC classification"
    },
    {
        "Dataset ID": "DS-CT-03",
        "Dataset Name": "AbdomenCT-1K",
        "Platform": "GitHub / Zenodo",
        "Primary Modality": "3D CT",
        "Target Organ Systems": "Abdominal System (Liver, Kidneys, Spleen, Pancreas)",
        "Anatomical Structures / Labels": "4 major abdominal organs under multi-center variations",
        "Sample Size / Scale": "1,000+ CT scans",
        "Primary Disease Taxonomy": "Neoplastic, Infectious, Vascular",
        "Target Medical Specialist": "Abdominal Radiologist, Oncologist",
        "Data Access / Download Link": "https://github.com/dwyang/AbdomenCT-1K",
        "Licensing": "Open Research License",
        "Format": "NIfTI (.nii.gz)",
        "Pre-training / Validation Role": "Abdominal multi-organ segmentation benchmarking"
    },
    {
        "Dataset ID": "DS-CT-04",
        "Dataset Name": "TCIA TCGA-KIRC (Renal CT)",
        "Platform": "The Cancer Imaging Archive (TCIA)",
        "Primary Modality": "3D Contrast CT",
        "Target Organ Systems": "Urinary & Renal System (Kidneys)",
        "Anatomical Structures / Labels": "Renal cortex, Medulla, Clear cell renal carcinoma lesions",
        "Sample Size / Scale": "488 volumetric series",
        "Primary Disease Taxonomy": "Neoplastic, Autoimmune, Vascular",
        "Target Medical Specialist": "Oncologist, Nephrologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/TCGA-KIRC",
        "Licensing": "TCIA Open Access Data Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Renal tumor characterization"
    },
    {
        "Dataset ID": "DS-CT-05",
        "Dataset Name": "Kaggle RSNA Pulmonary Embolism CT Challenge",
        "Platform": "Kaggle",
        "Primary Modality": "3D CT Angiography (CTA)",
        "Target Organ Systems": "Cardiovascular & Thoracic System",
        "Anatomical Structures / Labels": "Pulmonary vascular embolism location, RV/LV ratio, chronic PE",
        "Sample Size / Scale": "12,000+ CT scans",
        "Primary Disease Taxonomy": "Vascular, Acute Pulmonary",
        "Target Medical Specialist": "Cardiologist, Radiologist",
        "Data Access / Download Link": "https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection",
        "Licensing": "Kaggle Competition License",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Vascular PE embolus detection"
    },

    # --- 3D MRI DATASETS ---
    {
        "Dataset ID": "DS-MRI-01",
        "Dataset Name": "BraTS 2023 (Brain Tumor Segmentation)",
        "Platform": "Synapse",
        "Primary Modality": "3D Multi-Sequence MRI (T1, T1Gd, T2, FLAIR)",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Enhancing tumor, Non-enhancing core, Peritumoral edema",
        "Sample Size / Scale": "4,500 volumetric MRI scans",
        "Primary Disease Taxonomy": "Neoplastic, Neurodegenerative, Demyelinating",
        "Target Medical Specialist": "Neurologist, Neurosurgeon",
        "Data Access / Download Link": "https://www.synapse.org/#!Synapse:syn51156910/wiki/622351",
        "Licensing": "Synapse Research License",
        "Format": "NIfTI (.nii.gz)",
        "Pre-training / Validation Role": "3D neuro-oncology sub-region segmentation"
    },
    {
        "Dataset ID": "DS-MRI-02",
        "Dataset Name": "TCIA TCGA-LGG / TCGA-GBM",
        "Platform": "The Cancer Imaging Archive (TCIA)",
        "Primary Modality": "3D Multi-Parametric MRI",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Brain parenchyma, Glioma sub-regions (Enhancing, Edema, Core)",
        "Sample Size / Scale": "1,000+ volumetric MRI series",
        "Primary Disease Taxonomy": "Neoplastic, Demyelinating, Traumatic",
        "Target Medical Specialist": "Neurologist, Neurosurgeon, Oncologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG",
        "Licensing": "TCIA Open Access Policy",
        "Format": "DICOM / NIfTI",
        "Pre-training / Validation Role": "Brain soft-tissue tumor classification"
    },
    {
        "Dataset ID": "DS-MRI-03",
        "Dataset Name": "TCIA PROSTATEx",
        "Platform": "The Cancer Imaging Archive (TCIA)",
        "Primary Modality": "3D Multi-Parametric MRI (T2w, DCE, DWI)",
        "Target Organ Systems": "Pelvic & Reproductive System (Prostate)",
        "Anatomical Structures / Labels": "Prostate peripheral/transition zones, Clinically significant lesions",
        "Sample Size / Scale": "330 patient studies",
        "Primary Disease Taxonomy": "Neoplastic, Autoimmune",
        "Target Medical Specialist": "Oncologist, Radiologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges",
        "Licensing": "TCIA Open Access Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Pelvic multi-sequence lesion localization"
    },
    {
        "Dataset ID": "DS-MRI-04",
        "Dataset Name": "IXI Dataset (Information eXtraction from Images)",
        "Platform": "Brain-Development.org",
        "Primary Modality": "3D Brain MRI (T1, T2, PD, MRA, DTI)",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Normal healthy brain anatomical structures across subjects",
        "Sample Size / Scale": "nearly 600 MRI volumes",
        "Primary Disease Taxonomy": "Neurodegenerative Baseline",
        "Target Medical Specialist": "Neurologist, Neuroscientist",
        "Data Access / Download Link": "https://brain-development.org/ixi-dataset/",
        "Licensing": "CC BY-SA 3.0",
        "Format": "NIfTI (.nii.gz)",
        "Pre-training / Validation Role": "Normal brain anatomical representation pre-training"
    },
    {
        "Dataset ID": "DS-MRI-05",
        "Dataset Name": "OASIS-3 (Open Access Series of Imaging Studies)",
        "Platform": "Central Neuroimaging Data Archive (CNDA)",
        "Primary Modality": "3D Brain MRI & PET",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Alzheimer's disease biomarkers, Ventricular enlargement, Cortical thickness",
        "Sample Size / Scale": "1,098 participants (2,168 MR sessions)",
        "Primary Disease Taxonomy": "Neurodegenerative, Demyelinating",
        "Target Medical Specialist": "Neurologist, Geriatric Specialist",
        "Data Access / Download Link": "https://www.oasis-brains.org/",
        "Licensing": "OASIS Open Access Policy",
        "Format": "NIfTI / DICOM",
        "Pre-training / Validation Role": "Neurodegenerative brain atrophy benchmark"
    },

    # --- 2D X-RAY DATASETS ---
    {
        "Dataset ID": "DS-XRAY-01",
        "Dataset Name": "MIMIC-CXR-JPG (v2.0.0)",
        "Platform": "PhysioNet",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "14 pathology classes (Pneumonia, Effusion, Atelectasis, etc.)",
        "Sample Size / Scale": "377,110 X-rays (227,835 studies)",
        "Primary Disease Taxonomy": "Infectious, Neoplastic, Traumatic, Vascular",
        "Target Medical Specialist": "Pulmonologist, Infectious Disease Specialist",
        "Data Access / Download Link": "https://physionet.org/content/mimic-cxr-jpg/2.0.0/",
        "Licensing": "PhysioNet Credentialed License",
        "Format": "JPEG / DICOM",
        "Pre-training / Validation Role": "Thoracic projection pre-training & zero-shot screening"
    },
    {
        "Dataset ID": "DS-XRAY-02",
        "Dataset Name": "NIH ChestX-ray14",
        "Platform": "NIH Box / Kaggle",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "14 thoracic disease labels mined via NLP from reports",
        "Sample Size / Scale": "112,120 frontal X-rays (30,805 patients)",
        "Primary Disease Taxonomy": "Infectious, Neoplastic, Vascular",
        "Target Medical Specialist": "Radiologist, Pulmonologist",
        "Data Access / Download Link": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "Licensing": "Public Domain / NIH License",
        "Format": "PNG",
        "Pre-training / Validation Role": "2D projection disease classification benchmarking"
    },
    {
        "Dataset ID": "DS-XRAY-03",
        "Dataset Name": "CheXpert Dataset",
        "Platform": "Stanford ML Group",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "14 observation classes with uncertainty labels",
        "Sample Size / Scale": "224,316 chest radiographs (65,240 patients)",
        "Primary Disease Taxonomy": "Infectious, Traumatic, Vascular",
        "Target Medical Specialist": "Pulmonologist, Radiologist",
        "Data Access / Download Link": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "Licensing": "Stanford University Research License",
        "Format": "JPG",
        "Pre-training / Validation Role": "Multi-class thoracic screening benchmark"
    },
    {
        "Dataset ID": "DS-XRAY-04",
        "Dataset Name": "COVID-19 Radiography Database",
        "Platform": "Kaggle",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "COVID-19, Viral Pneumonia, Lung Opacity, Normal",
        "Sample Size / Scale": "21,165 X-ray images",
        "Primary Disease Taxonomy": "Infectious",
        "Target Medical Specialist": "Infectious Disease Specialist, Pulmonologist",
        "Data Access / Download Link": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        "Licensing": "CC BY 4.0",
        "Format": "PNG",
        "Pre-training / Validation Role": "Infectious viral pulmonary screening benchmark"
    },
    {
        "Dataset ID": "DS-XRAY-05",
        "Dataset Name": "VinDr-CXR (Vietnamese CXR)",
        "Platform": "PhysioNet / Kaggle",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "22 local lesions (bounding boxes) and 6 global diseases",
        "Sample Size / Scale": "18,000 DICOM images",
        "Primary Disease Taxonomy": "Infectious, Neoplastic, Traumatic",
        "Target Medical Specialist": "Radiologist, Pulmonologist",
        "Data Access / Download Link": "https://physionet.org/content/vindr-cxr/1.0.0/",
        "Licensing": "PhysioNet Open License",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Localized lesion detection benchmarking"
    },

    # --- 2D ULTRASOUND & ECHOCARDIOGRAPHY DATASETS ---
    {
        "Dataset ID": "DS-US-01",
        "Dataset Name": "BUSI (Breast Ultrasound Images Dataset)",
        "Platform": "Kaggle",
        "Primary Modality": "2D Breast Ultrasound",
        "Target Organ Systems": "Mammary & Soft Tissue System",
        "Anatomical Structures / Labels": "Normal, Benign, and Malignant breast lesions with masks",
        "Sample Size / Scale": "780 ultrasound images (600 patients)",
        "Primary Disease Taxonomy": "Neoplastic",
        "Target Medical Specialist": "Oncologist, Breast Radiologist",
        "Data Access / Download Link": "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset",
        "Licensing": "CC BY 4.0",
        "Format": "PNG",
        "Pre-training / Validation Role": "Ultrasound soft-tissue tumor classification"
    },
    {
        "Dataset ID": "DS-US-02",
        "Dataset Name": "CAMUS (Cardiac Acquisition for Multi-structure Ultrasound)",
        "Platform": "CREATIS Challenge",
        "Primary Modality": "2D Echocardiography (Cardiac Ultrasound)",
        "Target Organ Systems": "Cardiovascular System (Heart)",
        "Anatomical Structures / Labels": "Left ventricle endocardium/epicardium, Left atrium masks",
        "Sample Size / Scale": "500 clinical patient acquisitions",
        "Primary Disease Taxonomy": "Vascular, Structural Cardiac",
        "Target Medical Specialist": "Cardiologist, Vascular Surgeon",
        "Data Access / Download Link": "https://www.creatis.insa-lyon.fr/Challenge/camus/",
        "Licensing": "CREATIS Open Research License",
        "Format": "MHD / Raw",
        "Pre-training / Validation Role": "Echocardiogram cardiac chamber segmentation"
    },
    {
        "Dataset ID": "DS-US-03",
        "Dataset Name": "Ultrasound Nerve Segmentation Dataset",
        "Platform": "Kaggle",
        "Primary Modality": "2D Ultrasound",
        "Target Organ Systems": "Peripheral Nervous System (Neck / Brachial Plexus)",
        "Anatomical Structures / Labels": "Brachial Plexus nerve structures pixel-level masks",
        "Sample Size / Scale": "5,638 ultrasound images",
        "Primary Disease Taxonomy": "Traumatic, Peripheral Nervous",
        "Target Medical Specialist": "Neurologist, Anesthesiologist",
        "Data Access / Download Link": "https://www.kaggle.com/c/ultrasound-nerve-segmentation",
        "Licensing": "Kaggle Competition License",
        "Format": "JPEG",
        "Pre-training / Validation Role": "Peripheral nerve segmentation benchmark"
    },

    # --- 2D MAMMOGRAPHY DATASETS ---
    {
        "Dataset ID": "DS-MAMMO-01",
        "Dataset Name": "CBIS-DDSM (Curated Breast Imaging DDSM)",
        "Platform": "The Cancer Imaging Archive (TCIA)",
        "Primary Modality": "2D Digital Mammography",
        "Target Organ Systems": "Mammary Glandular System",
        "Anatomical Structures / Labels": "Calcifications, Masses, ROI bounding boxes, Pathology labels",
        "Sample Size / Scale": "1,566 cases (6,775 mammography images)",
        "Primary Disease Taxonomy": "Neoplastic",
        "Target Medical Specialist": "Oncologist, Mammographer",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM",
        "Licensing": "TCIA Open Access Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Mammographic micro-calcification detection"
    },
    {
        "Dataset ID": "DS-MAMMO-02",
        "Dataset Name": "InBreast Mammography Dataset",
        "Platform": "Mendeley Data",
        "Primary Modality": "2D Full-Field Digital Mammography (FFDM)",
        "Target Organ Systems": "Mammary Glandular System",
        "Anatomical Structures / Labels": "Masses, Calcifications, Architectural distortions, Spicules",
        "Sample Size / Scale": "115 cases (410 images)",
        "Primary Disease Taxonomy": "Neoplastic",
        "Target Medical Specialist": "Oncologist, Radiologist",
        "Data Access / Download Link": "https://data.mendeley.com/datasets/ywsfp3v2bc/1",
        "Licensing": "Open Access Research License",
        "Format": "DICOM",
        "Pre-training / Validation Role": "High-resolution FFDM lesion classification"
    },

    # --- 3D HYBRID PET-CT DATASET ---
    {
        "Dataset ID": "DS-PET-01",
        "Dataset Name": "TCIA FDG-PET-CT Lesion Dataset (AutoPET)",
        "Platform": "The Cancer Imaging Archive (TCIA)",
        "Primary Modality": "3D Hybrid PET-CT",
        "Target Organ Systems": "Whole Body / Systemic",
        "Anatomical Structures / Labels": "Metabolically active tumor lesions, SUV uptake maps",
        "Sample Size / Scale": "1,014 PET-CT studies",
        "Primary Disease Taxonomy": "Neoplastic, Metastatic",
        "Target Medical Specialist": "Oncologist, Nuclear Medicine Specialist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/FDG-PET-CT-Lesions",
        "Licensing": "TCIA Open Access Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Metabolic multi-organ tumor localization"
    },

    # --- 2D HISTOPATHOLOGY (WSI) DATASETS ---
    {
        "Dataset ID": "DS-HIST-01",
        "Dataset Name": "CAMELYON16 / CAMELYON17",
        "Platform": "Grand Challenge",
        "Primary Modality": "2D Whole Slide Histopathology (WSI)",
        "Target Organ Systems": "Lymphatic System (Axillary Lymph Nodes)",
        "Anatomical Structures / Labels": "Breast cancer lymph node metastasis annotations",
        "Sample Size / Scale": "1,000 WSI slides",
        "Primary Disease Taxonomy": "Neoplastic, Metastatic",
        "Target Medical Specialist": "Pathologist, Surgical Oncologist",
        "Data Access / Download Link": "https://camelyon17.grand-challenge.org/",
        "Licensing": "CC0 / Public Benchmark",
        "Format": "TIF / TFF (Gigapixel WSI)",
        "Pre-training / Validation Role": "Microscopic cellular metastatic tissue analysis"
    },
    {
        "Dataset ID": "DS-HIST-02",
        "Dataset Name": "PANDA (Prostate cANCer Grade Assessment)",
        "Platform": "Kaggle",
        "Primary Modality": "2D Whole Slide Histopathology (WSI)",
        "Target Organ Systems": "Pelvic & Reproductive System (Prostate)",
        "Anatomical Structures / Labels": "ISUP grade tissue masks, Gleason pattern annotations",
        "Sample Size / Scale": "11,000 WSI slides",
        "Primary Disease Taxonomy": "Neoplastic",
        "Target Medical Specialist": "Pathologist, Urologist",
        "Data Access / Download Link": "https://www.kaggle.com/c/prostate-cancer-grade-assessment",
        "Licensing": "Kaggle Competition License",
        "Format": "TIFF",
        "Pre-training / Validation Role": "Histological cancer grading benchmark"
    }
]

def generate_csv():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ultimate_dataset_catalog[0].keys())

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ultimate_dataset_catalog:
            writer.writerow(row)

    print(f"Dataset Catalog CSV generated successfully at: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_csv()
