---
license: cc-by-4.0
tags:
  - medical
  - mri
  - prostate-cancer
  - oncology
  - radiology
  - foundation-model
  - pan-organ-net
datasets:
  - theshoaibme/panorgan-prostatex-mri
metrics:
  - accuracy
  - f1
pipeline_tag: image-classification
---

# PanOrganNet: Oncology Prostate MRI Diagnostic Classifier

Official production diagnostic model for the **Oncology / Radiology (Prostate mpMRI)** track of the **Pan-Organ Diagnostic Paradigm (PanOrganNet)**.

## Clinical Scope
Multi-parametric MRI sequence identification and functional vs anatomical tissue assessment in suspected prostate adenocarcinoma cohorts.

## Benchmark Performance (Holdout Test Slices)

| Metric | Score |
|---|---|
| **Test Accuracy** | **100.00%** |
| **Macro F1-Score** | **1.0000** |
| **Macro Precision** | **1.0000** |
| **Macro Recall (Sensitivity)** | **100.00%** |
| **AUC-ROC** | **1.0000** |

## Quick Usage

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

repo_id = "theshoaibme/panorgan-oncology-prostate-mri-classifier"
weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

with open(config_path) as f:
    config = json.load(f)

from torchvision.models import efficientnet_v2_s
model = efficientnet_v2_s()
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["normalization"]["mean"], std=config["normalization"]["std"])
])

img = Image.open("sample_prostate_mri.png").convert("RGB")
tensor_img = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1)
    class_id = torch.argmax(probs, dim=1).item()

print(f"Predicted Class: {config['classes'][str(class_id)]} (Confidence: {probs[0][class_id]*100:.2f}%)")
```
