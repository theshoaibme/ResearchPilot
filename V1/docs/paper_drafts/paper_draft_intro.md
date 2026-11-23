# The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening

## Abstract

Clinical Artificial Intelligence (AI) stands at a critical juncture: while deep neural networks demonstrate remarkable accuracy in isolated tasks, their real-world utility is heavily bottlenecked by **model fragmentation**—the classic "One Model, One Task" paradigm. Deploying, managing, and maintaining dozens of distinct deep networks for individual anatomical organs and scanner modalities is clinically and computationally unsustainable. 

In this paper, we introduce **Pan-Organ Net**, a high-capacity volumetric foundation model designed to establish a unified diagnostic paradigm across diverse human organs (including brain, lung, liver, heart, and kidneys) and multiple imaging modalities (CT, MRI, Ultrasound, and X-Ray). By combining a modality-aware anatomical tokenization scheme with a **Saliency-Guided Masked Autoencoder (MAE)** framework, our approach prevents dominant-organ shortcut learning under simulated Missing Not at Random (MNAR) data distributions. 

Furthermore, we bridge the clinical "Actionability Gap" by integrating a downstream recommendation head that translates latent anatomical embeddings into actionable diagnostic pathways, moving medical imaging AI from reactive classification toward proactive decision support.

---

## I. Introduction

Modern healthcare systems rely on a complex array of imaging technologies to diagnose, stage, and monitor diseases. Within any single clinical path, a patient might undergo a chest X-ray for immediate screening, a follow-up computed tomography (CT) scan for structural evaluation, and a high-resolution magnetic resonance imaging (MRI) scan to characterize soft tissue boundaries. Despite the inherent anatomical and physiological intersections across these examinations, modern medical computer vision models process them in absolute isolation. 

This status quo is defined by task-specific architectures: one convolutional neural network segments the liver, another classifies lung nodules, and a third detects cardiac chamber anomalies. In clinical environments, this architectural division leads to severe software fragmentation, astronomical computational overheads in hospital data centers, and an inability to recognize systemic pathology that crosses organ systems, such as metastatic spread, systemic inflammatory states, or vascular degradation.

To address these limitations, we propose the **Pan-Organ Diagnostic Paradigm**, materialized through **Pan-Organ Net**. This work shifts the baseline of medical AI from narrow task specialists to a generalist foundation model that maps the entire human body across different imaging modalities to a shared, high-dimensional latent space. 

Our core contribution lies in three main design elements:
1. **Modality-Aware Tokenization:** A patch projection layers that accepts heterogeneous inputs (2D/3D, varying pixel resolutions, voxel spacing, and acquisition physics) and maps them to unified feature dimensions.
2. **Saliency-Guided Masking (SGM):** An unsupervised pretext task that forces the network to learn holistic anatomical relationships, specifically countering the tendency of models to overfit to dominant, high-contrast organs (e.g., bones or blood vessels) while ignoring subtle pathological changes in smaller glands or soft tissue.
3. **Decoupled Actionable Heads:** Embedding decoder blocks that do not merely predict classifications but recommend next-step clinical workflows, thereby addressing the clinical translation gap where radiologists require actionable pathways rather than raw probability scores.

Through evaluation on a heterogeneous dataset compiling MIMIC-CXR, TotalSegmentator, and the Cancer Imaging Archive (TCIA), we demonstrate that Pan-Organ Net achieves superior zero-shot and few-shot classification and segmentation capabilities compared to existing state-of-the-art models, while demonstrating high robustness in simulated clinical scenarios with incomplete or missing scan series.
