## II. Related Work

### 2.1 Task-Specific Medical Architectures and Domain Shift
Historically, medical image analysis has been dominated by highly tailored deep architectures. Convolutional neural networks (CNNs), such as the U-Net and its volumetric extension, the 3D U-Net, established early benchmarks in medical organ segmentation. Although these networks perform exceptionally well when validated on homogeneous validation sets, they suffer from severe performance drops when deployed across different clinical sites. 

This susceptibility to **domain shift**—typically driven by differences in slice thickness, scanner manufacturer acquisition settings, and physical noise profiles—has motivated significant research into domain adaptation. However, these techniques remain reactive, seeking to align features *after* a model has been trained on a narrow task rather than building generic, robust representation spaces from the ground up.

### 2.2 Self-Supervised Learning and Vision Transformers in Medicine
To overcome the limitations of labeled medical data, the paradigm of Self-Supervised Learning (SSL) has gained significant momentum. Masked Image Modeling (MIM), popularized by the Masked Autoencoder (MAE), has emerged as a powerful pretext task. In medical imaging, models like *RadImageNet* and *BiomedCLIP* demonstrate that pre-training on massive datasets enables robust feature extraction that transfers well to downstream medical tasks. 

More recently, the transition from CNNs to Vision Transformers (ViTs) has allowed models to leverage self-attention mechanisms to capture long-range spatial and anatomical dependencies. Nevertheless, existing biomedical foundation models are predominantly 2D-centric or restricted to a single modality (such as chest X-rays or brain MRIs), limiting their utility in clinical situations requiring holistic, 3D anatomical evaluations.

### 2.3 Body-wide and Multi-organ Frameworks
A concurrent direction in medical AI focuses on parsing the entire human body. Frameworks such as *TotalSegmentator* have shown the capability of segmenting over a hundred structures from a single CT scan. 

However, multi-organ models typically require explicit manual segmentation annotations for every target structure, which is highly labor-intensive. In contrast, emerging generalist architectures, such as *Pan-FM*, utilize self-supervised pre-training to learn multi-organ representations from large unlabeled volumetric datasets. 

While these models represent a significant step forward, they are often prone to "dominant-organ shortcutting" under real-world, missing-organ scenarios (Missing Not at Random distributions). In this work, we propose **Saliency-Guided Masking (SGM)** to force the model to capture uniform cross-organ feature spaces without relying heavily on high-contrast anatomical shortcuts.
