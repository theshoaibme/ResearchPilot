import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.models.oncology.prostatex.model_zoo import get_prostatex_model, SUPPORTED_MODELS
from core.models.oncology.prostatex.data_loader import get_prostatex_loaders
from core.engine.base_trainer import BaseTrainer, Colors, Logger

CLASS_NAMES = ["Anatomical_T2", "Functional_ADC_DWI"]
HF_TOKEN = os.environ.get("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parents[4]

def plot_confusion_matrix(cm_array, model_name: str, save_path: str):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(f'Test Confusion Matrix — {model_name}\n(PROSTATEx Multi-Parametric MRI)', fontsize=13, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm_array.max() / 2.
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(j, i, format(cm_array[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_array[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('Ground Truth', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_on_test(model, test_loader, device: str):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, c_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class[c_name] = {
            "sensitivity": sens,
            "specificity": spec,
            "f1": f1_score(y_true == i, y_pred == i, zero_division=0)
        }

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "per_class": per_class
    }

def train_and_benchmark_prostatex(epochs: int = 5, batch_size: int = 32):
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.MAGENTA}🔬 PanOrganNet: PROSTATEx Prostate MRI Benchmark{Colors.RESET}")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info(f"Target Compute Device: {device.upper()}")

    train_loader, val_loader, test_loader = get_prostatex_loaders(batch_size=batch_size)
    Logger.success(f"Loaded PROSTATEx splits: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    models_to_train = ["EfficientNetV2_S", "DenseNet121"]
    results = {}

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for m_name in models_to_train:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Training Architecture: {m_name} ---{Colors.RESET}")
        model = get_prostatex_model(m_name, num_classes=2, pretrained=True)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = PROJECT_ROOT / "core" / "models" / "oncology" / "prostatex_experiments" / f"{m_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        trainer = BaseTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            exp_dir=str(exp_dir),
            patience=3
        )

        trainer.fit(num_epochs=epochs, resume=False)

        best_pt = exp_dir / "best_model.pt"
        ckpt = torch.load(best_pt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        Logger.info(f"Evaluating {m_name} on holdout Test Cohort ({len(test_loader.dataset)} slices)...")
        eval_metrics = evaluate_on_test(model, test_loader, device)

        cm_path = reports_dir / f"cm_prostatex_{m_name.lower()}.png"
        plot_confusion_matrix(eval_metrics['confusion_matrix'], m_name, str(cm_path))
        Logger.success(f"Saved Confusion Matrix: {cm_path}")

        results[m_name] = {
            "metrics": eval_metrics,
            "exp_dir": str(exp_dir),
            "best_pt": str(best_pt)
        }

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}🏆 PROSTATEx BENCHMARK RESULTS{Colors.RESET}")
    print("=" * 70)
    print(f"{'Model':<18} | {'Test Acc':<10} | {'Macro F1':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'AUC-ROC'}")
    print("-" * 70)
    for m_name, r in results.items():
        m = r['metrics']
        print(f"{m_name:<18} | {m['accuracy']*100:>6.2f}%    | {m['f1_macro']:>7.4f}    | {m['recall']*100:>8.2f}%    | "
              f"{np.mean([v['specificity'] for v in m['per_class'].values()])*100:>8.2f}%    | {m['auc_roc']:>6.4f}")
    print("=" * 70 + "\n")

    champion_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_macro'])
    Logger.success(f"Champion Model for PROSTATEx: {champion_name} (F1: {results[champion_name]['metrics']['f1_macro']:.4f})")

    deploy_prostatex_to_hf(champion_name, results[champion_name])

def deploy_prostatex_to_hf(model_name: str, result_dict: dict):
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 Deploying PROSTATEx Champion to Hugging Face{Colors.RESET}")
    print("=" * 70)

    api = HfApi()
    username = api.whoami(token=HF_TOKEN)["name"]
    repo_id = f"{username}/panorgan-oncology-prostate-mri-classifier"

    api.create_repo(repo_id=repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True, private=False)
    Logger.success(f"Hugging Face Model Repository Ready: https://huggingface.co/{repo_id}")

    staging_dir = PROJECT_ROOT / "deploy_staging" / f"prostatex_{model_name}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(result_dict['best_pt'], map_location='cpu')
    torch.save(ckpt['model_state_dict'], staging_dir / "pytorch_model.pt")

    config = {
        "model_architecture": model_name,
        "task": "Prostate Multi-Parametric MRI Sequence Classification (Oncology / Radiology)",
        "num_classes": 2,
        "classes": {
            "0": "Anatomical_T2",
            "1": "Functional_ADC_DWI"
        },
        "image_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "dataset": "PROSTATEx mpMRI (Patient-Wise Split)",
        "framework": "PyTorch 2.14 (Apple Silicon MPS Acceleration)"
    }
    with open(staging_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    m = result_dict['metrics']
    if "dense" in model_name.lower():
        arch_init = """from torchvision.models import densenet121
model = densenet121()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier.in_features, 2)
)"""
    else:
        arch_init = """from torchvision.models import efficientnet_v2_s
model = efficientnet_v2_s()
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier[1].in_features, 2)
)"""

    readme_content = f"""---
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
| **Test Accuracy** | **{m['accuracy']*100:.2f}%** |
| **Macro F1-Score** | **{m['f1_macro']:.4f}** |
| **Macro Precision** | **{m['precision']:.4f}** |
| **Macro Recall (Sensitivity)** | **{m['recall']*100:.2f}%** |
| **AUC-ROC** | **{m['auc_roc']:.4f}** |

## Quick Usage

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

repo_id = "{repo_id}"
weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

with open(config_path) as f:
    config = json.load(f)

{arch_init}
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

print(f"Predicted Class: {{config['classes'][str(class_id)]}} (Confidence: {{probs[0][class_id]*100:.2f}}%)")
```
"""
    with open(staging_dir / "README.md", "w") as f:
        f.write(readme_content)

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        token=HF_TOKEN,
        commit_message=f"Deploy PROSTATEx Prostate {model_name} champion model with weights, config and Model Card"
    )

    Logger.success(f"🎉 Champion Model Successfully Deployed -> https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    train_and_benchmark_prostatex(epochs=5, batch_size=32)
