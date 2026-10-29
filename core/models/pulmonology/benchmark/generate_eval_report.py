import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.models.pulmonology.benchmark.model_zoo import get_model, count_parameters
from core.models.pulmonology.benchmark.data_loader import get_pulmonology_loaders
from core.engine.base_trainer import Colors, Logger

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    import seaborn as sns
    HAS_EVAL_LIBS = True
except ImportError:
    HAS_EVAL_LIBS = False

CLASS_NAMES = ["Normal", "COVID", "Lung_Opacity", "Viral_Pneumonia"]

def evaluate_checkpoint(model_name: str, checkpoint_path: str, test_loader, device: str):
    Logger.info(f"Evaluating {model_name} from: {checkpoint_path}")
    
    model = get_model(model_name, num_classes=4, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(test_loader, desc=f"Testing {model_name}", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
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

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class sensitivity and specificity
    per_class_metrics = {}
    for i, c_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_metrics[c_name] = {
            "sensitivity": sens,
            "specificity": spec,
            "f1": f1_score(y_true == i, y_pred == i, zero_division=0)
        }

    return {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "auc_roc": auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics
    }

def plot_confusion_matrix(cm_list, model_name: str, save_path: str):
    cm = np.array(cm_list)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=30)
    plt.yticks(tick_marks, CLASS_NAMES)

    # Normalize values for text overlay
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True Clinical Category', fontsize=12)
    plt.xlabel('Predicted Diagnosis', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_full_report():
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}📊 Generating PanOrganNet Pulmonology Test Benchmark Report{Colors.RESET}")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info(f"Using Compute Device: {device.upper()}")

    _, _, test_loader = get_pulmonology_loaders(batch_size=32)
    Logger.info(f"Loaded Test Cohort: {len(test_loader.dataset):,} holdout images.")

    # Find the latest best checkpoints in experiments directory
    exp_root = Path("core/models/pulmonology/experiments")
    if not exp_root.exists():
        Logger.error("No experiments folder found.")
        return

    # Find best model for each architecture
    candidates = {}
    for d in exp_root.iterdir():
        if d.is_dir():
            best_pt = d / "best_model.pt"
            if best_pt.exists():
                name_prefix = d.name.split('_202')[0]
                # store the latest
                candidates[name_prefix] = best_pt

    if not candidates:
        Logger.warn("No trained best_model.pt checkpoints found in experiments.")
        return

    results = []
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    for m_name, ckpt in candidates.items():
        res = evaluate_checkpoint(m_name, str(ckpt), test_loader, device)
        results.append(res)
        
        # Save individual confusion matrix
        cm_path = reports_dir / f"cm_{m_name.lower()}.png"
        plot_confusion_matrix(res["confusion_matrix"], m_name, str(cm_path))
        Logger.success(f"Saved Confusion Matrix for {m_name} -> {cm_path}")

    # Build Markdown Comparison Table
    table_lines = [
        "# PanOrganNet: Pulmonology Multi-Model Benchmark Test Report",
        "",
        f"**Test Cohort**: 3,175 independent holdout Chest X-Rays (Stratified 15% Split)",
        f"**Evaluation Date**: {Path.cwd()}",
        "",
        "## Overall Comparative Performance",
        "",
        "| Architecture | Parameters (M) | Test Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | AUC-ROC |",
        "|---|---|---|---|---|---|---|"
    ]

    for r in sorted(results, key=lambda x: x['f1_macro'], reverse=True):
        m = get_model(r['model_name'], num_classes=4, pretrained=False)
        p = count_parameters(m)['total_params_million']
        table_lines.append(
            f"| **{r['model_name']}** | {p}M | "
            f"**{r['accuracy']*100:.2f}%** | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | **{r['f1_macro']:.4f}** | {r['auc_roc']:.4f} |"
        )

    table_lines.extend([
        "",
        "## Per-Class Sensitivity & Specificity Breakdown",
        "",
        "| Architecture | Class | Sensitivity (Recall) | Specificity | Class F1 |",
        "|---|---|---|---|---|"
    ])

    for r in results:
        for c_name in CLASS_NAMES:
            c_data = r['per_class'][c_name]
            table_lines.append(
                f"| {r['model_name']} | {c_name} | {c_data['sensitivity']*100:.2f}% | "
                f"{c_data['specificity']*100:.2f}% | {c_data['f1']:.4f} |"
            )

    report_md_path = reports_dir / "pulmonology_benchmark_report.md"
    report_md_path.write_text("\n".join(table_lines))
    Logger.success(f"Publication Benchmark Report saved -> {report_md_path}")

    print("\n" + report_md_path.read_text())

if __name__ == "__main__":
    generate_full_report()
