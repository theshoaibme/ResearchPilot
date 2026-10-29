---
license: cc-by-4.0
tags:
  - medical
  - breast-ultrasound
  - oncology
  - cancer-detection
  - foundation-model
  - pan-organ-net
datasets:
  - theshoaibme/panorgan-busi-breast-ultrasound
metrics:
  - accuracy
  - f1
pipeline_tag: image-classification
---

# PanOrganNet: Oncology Breast Ultrasound Diagnostic Classifier

Official production diagnostic model for the **Oncology (Breast Ultrasound)** track of the **Pan-Organ Diagnostic Paradigm (PanOrganNet)**.

## Clinical Scope
Automated multi-class classification of breast lesions on 2D ultrasound scans:
- **Normal**: Normal breast tissue with no identifiable focal masses.
- **Benign**: Fibroadenomas, benign cysts, and non-cancerous masses.
- **Malignant**: Invasive ductal carcinoma, lobular carcinoma, and aggressive lesions requiring biopsy.

## Benchmark Performance (117 Holdout Ultrasound Test Scans)

| Metric | Score |
|---|---|
| **Test Accuracy** | **87.18%** |
| **Macro F1-Score** | **0.8622** |
| **Macro Precision** | **0.8462** |
| **Macro Recall (Sensitivity)** | **88.41%** |
| **AUC-ROC** | **0.9335** |

## Quick Usage

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

repo_id = "theshoaibme/panorgan-oncology-breast-ultrasound-classifier"
weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

with open(config_path) as f:
    config = json.load(f)

from torchvision.models import efficientnet_v2_s
model = efficientnet_v2_s()
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier[1].in_features, 3)
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["normalization"]["mean"], std=config["normalization"]["std"])
])

img = Image.open("sample_breast_ultrasound.png").convert("RGB")
tensor_img = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1)
    class_id = torch.argmax(probs, dim=1).item()

print(f"Predicted Pathology: {config['classes'][str(class_id)]} (Confidence: {probs[0][class_id]*100:.2f}%)")
```
