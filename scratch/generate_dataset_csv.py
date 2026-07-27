import csv
import os

# Define dataset catalog with direct download / access URLs
datasets = [
    {
        "Dataset ID": "DS-001",
        "Dataset Name": "TotalSegmentator (v2)",
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
        "Dataset ID": "DS-002",
        "Dataset Name": "MIMIC-CXR-JPG (v2.0.0)",
        "Primary Modality": "2D Chest X-Ray",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "14 pathology classes (Pneumonia, Cardiomegaly, Effusion, Atelectasis, etc.)",
        "Sample Size / Scale": "377,110 X-rays (227,835 studies)",
        "Primary Disease Taxonomy": "Infectious, Neoplastic, Traumatic, Vascular",
        "Target Medical Specialist": "Pulmonologist, Infectious Disease Specialist",
        "Data Access / Download Link": "https://physionet.org/content/mimic-cxr-jpg/2.0.0/",
        "Licensing": "PhysioNet Credentialed Data Use Agreement",
        "Format": "JPEG / DICOM",
        "Pre-training / Validation Role": "Thoracic projection pre-training & zero-shot screening"
    },
    {
        "Dataset ID": "DS-003",
        "Dataset Name": "TCIA TCGA-LGG / TCGA-GBM",
        "Primary Modality": "3D Multi-Parametric MRI",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Brain parenchyma, Glioma sub-regions (Enhancing, Edema, Core)",
        "Sample Size / Scale": "1,000+ volumetric MRI series",
        "Primary Disease Taxonomy": "Neoplastic, Demyelinating, Traumatic",
        "Target Medical Specialist": "Neurologist, Neurosurgeon, Oncologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG",
        "Licensing": "TCIA Open Access Data Policy",
        "Format": "DICOM / NIfTI",
        "Pre-training / Validation Role": "Neuro-oncology multi-parametric soft-tissue classification"
    },
    {
        "Dataset ID": "DS-004",
        "Dataset Name": "TCIA TCGA-KIRC",
        "Primary Modality": "3D Contrast CT & MRI",
        "Target Organ Systems": "Urinary & Renal System (Kidneys)",
        "Anatomical Structures / Labels": "Renal cortex, Medulla, Clear cell renal carcinoma lesions",
        "Sample Size / Scale": "488 CT/MRI volumetric series",
        "Primary Disease Taxonomy": "Neoplastic, Autoimmune, Vascular",
        "Target Medical Specialist": "Oncologist, Nephrologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/TCGA-KIRC",
        "Licensing": "TCIA Open Access Data Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Abdominal renal tumor characterization"
    },
    {
        "Dataset ID": "DS-005",
        "Dataset Name": "BraTS 2023 (Brain Tumor Segmentation)",
        "Primary Modality": "3D Multi-Sequence MRI (T1, T1Gd, T2, FLAIR)",
        "Target Organ Systems": "Central Nervous System (Brain)",
        "Anatomical Structures / Labels": "Enhancing tumor, Non-enhancing core, Peritumoral edema",
        "Sample Size / Scale": "4,500 volumetric MRI scans",
        "Primary Disease Taxonomy": "Neoplastic, Neurodegenerative, Demyelinating",
        "Target Medical Specialist": "Neurologist, Neurosurgeon",
        "Data Access / Download Link": "https://www.synapse.org/#!Synapse:syn51156910/wiki/622351",
        "Licensing": "Synapse Research License",
        "Format": "NIfTI (.nii.gz)",
        "Pre-training / Validation Role": "Fine-grained 3D brain lesion segmentation benchmark"
    },
    {
        "Dataset ID": "DS-006",
        "Dataset Name": "LUNA16 / LIDC-IDRI",
        "Primary Modality": "3D Low-Dose Chest CT",
        "Target Organ Systems": "Thoracic & Respiratory System",
        "Anatomical Structures / Labels": "1,018 lung nodule ground-truth annotations with malignancy scores",
        "Sample Size / Scale": "888 CT scans",
        "Primary Disease Taxonomy": "Neoplastic, Infectious, Traumatic",
        "Target Medical Specialist": "Pulmonologist, Radiologist",
        "Data Access / Download Link": "https://luna16.grand-challenge.org/",
        "Licensing": "Public Benchmark License",
        "Format": "MHD / Raw / NIfTI",
        "Pre-training / Validation Role": "Pulmonary nodule detection & AUC-ROC classification"
    },
    {
        "Dataset ID": "DS-007",
        "Dataset Name": "TCIA PROSTATEx",
        "Primary Modality": "3D Multi-Parametric MRI (T2w, DCE, DWI)",
        "Target Organ Systems": "Pelvic & Reproductive System",
        "Anatomical Structures / Labels": "Prostate peripheral/transition zones, Clinically significant lesions",
        "Sample Size / Scale": "330 patient studies",
        "Primary Disease Taxonomy": "Neoplastic, Autoimmune",
        "Target Medical Specialist": "Oncologist, Radiologist",
        "Data Access / Download Link": "https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges",
        "Licensing": "TCIA Open Access Data Policy",
        "Format": "DICOM",
        "Pre-training / Validation Role": "Pelvic multi-parametric MRI lesion localization"
    }
]

output_path = r"c:\Users\HP\OneDrive\Desktop\ResearchPilot\phases\Pan_Organ_Medical_Datasets_Catalog.csv"

headers = list(datasets[0].keys())

with open(output_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(datasets)

print(f"Dataset catalog successfully generated at: {output_path}")
