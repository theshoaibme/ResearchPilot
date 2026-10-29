---
license: cc-by-4.0
tags:
  - medical
  - chest-xray
  - pulmonology
  - covid-19
  - pneumonia
  - foundation-model
  - pan-organ-net
datasets:
  - theshoaibme/panorgan-processed-pulmonology
metrics:
  - accuracy
  - f1
pipeline_tag: image-classification
---

# PanOrganNet: Pulmonology Chest X-Ray Diagnostic Classifier

Official production diagnostic model for the **Pulmonology** track of the **Pan-Organ Diagnostic Paradigm (PanOrganNet)**.

## Clinical Scope
This model provides automated, multi-class screening across four pulmonary clinical categories:
- **Normal**: Healthy lungs with no visible pathological opacities.
- **COVID-19**: Severe acute respiratory syndrome coronavirus 2 presenting peripheral ground-glass opacities.
- **Lung Opacity**: Non-COVID pulmonary consolidations, bacterial infiltrates, and related opacities.
- **Viral Pneumonia**: Non-COVID interstitial viral infiltrates.

## Model Summary
- **Architecture**: `EfficientNetV2_S`
- **Parameters**: `20.18M` (20,182,612 parameters)
- **Input Resolution**: `224x224x3` (RGB normalized with standard medical ImageNet statistics)
- **Training Dataset**: [PanOrganNet Processed Pulmonology](https://huggingface.co/datasets/theshoaibme/panorgan-processed-pulmonology) (14,815 training radiographs, 3,175 validation radiographs)
- **Evaluation Benchmark**: 3,175 independent holdout radiographs (15% stratified test cohort)

## Benchmark Performance

| Evaluation Metric | Score |
|---|---|
| **Test Accuracy** | **95.87%** |
| **Macro F1-Score** | **0.9641** |
| **Sensitivity (Recall)** | **95.53%** |
| **Specificity** | **98.62%** |

## Quick Usage

### PyTorch Inference

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

# 1. Download Model Config & Weights
repo_id = "theshoaibme/panorgan-pulmonology-chestxray-classifier"
weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

with open(config_path) as f:
    config = json.load(f)

# 2. Recreate Architecture & Load Weights
from torchvision.models import efficientnet_v2_s
model = efficientnet_v2_s()
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.classifier[1].in_features, 4)
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# 3. Preprocess Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["normalization"]["mean"], std=config["normalization"]["std"])
])

img = Image.open("sample_chest_xray.png").convert("RGB")
tensor_img = transform(img).unsqueeze(0)

# 4. Predict
with torch.no_grad():
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1)
    class_id = torch.argmax(probs, dim=1).item()

print(f"Predicted Diagnosis: {config['classes'][str(class_id)]} (Confidence: {probs[0][class_id]*100:.2f}%)")
```

### Citation

If you use this model in your research, please cite:
```bibtex
@article{chowdhury2020can,
  title={Can AI help in screening Viral and COVID-19 pneumonia?},
  author={Chowdhury, MEH and Rahman, T and Khandakar, A and others},
  journal={IEEE Access},
  volume={8},
  pages={132665--132676},
  year={2020}
}
```
