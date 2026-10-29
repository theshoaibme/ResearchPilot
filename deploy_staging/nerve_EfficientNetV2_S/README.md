---
license: cc-by-4.0
tags:
  - medical
  - ultrasound
  - neurology
  - nerve-detection
  - foundation-model
  - pan-organ-net
datasets:
  - theshoaibme/panorgan-ultrasound-nerve-segmentation
metrics:
  - accuracy
  - f1
pipeline_tag: image-classification
---

# PanOrganNet: Neurology Ultrasound Nerve Diagnostic Classifier

Official production diagnostic model for the **Neurology (Brachial Plexus Ultrasound)** track of the **Pan-Organ Diagnostic Paradigm (PanOrganNet)**.

## Clinical Scope
Automated real-time screening and localization of the brachial plexus nerve cluster in ultrasound video/frames to guide anesthesiologists during catheter insertion.

## Benchmark Performance (959 Holdout Ultrasound Test Scans)

| Metric | Score |
|---|---|
| **Test Accuracy** | **77.06%** |
| **Macro F1-Score** | **0.7697** |
| **Macro Precision** | **0.7784** |
| **Macro Recall (Sensitivity)** | **77.25%** |
| **AUC-ROC** | **0.8662** |

## Quick Usage

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

repo_id = "theshoaibme/panorgan-neurology-nerve-ultrasound-classifier"
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

img = Image.open("sample_nerve_ultrasound.png").convert("RGB")
tensor_img = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1)
    class_id = torch.argmax(probs, dim=1).item()

print(f"Predicted Class: {config['classes'][str(class_id)]} (Confidence: {probs[0][class_id]*100:.2f}%)")
```
