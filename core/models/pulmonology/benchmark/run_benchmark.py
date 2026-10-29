import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.models.pulmonology.benchmark.model_zoo import get_model, count_parameters, SUPPORTED_MODELS
from core.models.pulmonology.benchmark.data_loader import get_pulmonology_loaders
from core.models.pulmonology.pipeline_2.loss_engineering import get_loss_criterion
from core.engine.base_trainer import BaseTrainer, Colors, Logger

def print_banner(model_name: str, device: str, params: dict, epochs: int, batch_size: int, loss_type: str):
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.MAGENTA}🫁 PanOrganNet: Pulmonology S.O.T.A. Benchmark Training{Colors.RESET}")
    print("=" * 70)
    print(f"  {Colors.CYAN}Model Architecture:{Colors.RESET} {Colors.BOLD}{model_name}{Colors.RESET}")
    print(f"  {Colors.CYAN}Total Parameters:{Colors.RESET}   {params['total_params_million']} Million ({params['total_parameters']:,})")
    print(f"  {Colors.CYAN}Target Classes:{Colors.RESET}     4 (Normal, COVID-19, Lung Opacity, Viral Pneumonia)")
    print(f"  {Colors.CYAN}Compute Device:{Colors.RESET}     {Colors.GREEN}{device.upper()}{Colors.RESET}")
    print(f"  {Colors.CYAN}Epochs / Batch:{Colors.RESET}     {epochs} epochs | Batch Size: {batch_size}")
    print(f"  {Colors.CYAN}Loss Function:{Colors.RESET}      {loss_type.upper()}")
    print("=" * 70 + "\n")

def train_single_model(
    model_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    loss_type: str = "focal",
    patience: int = 4,
    device: str = None
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # 1. Initialize Model
    model = get_model(model_name, num_classes=4, pretrained=True)
    params = count_parameters(model)
    print_banner(model_name, device, params, epochs, batch_size, loss_type)

    # 2. Load Data
    Logger.info(f"Loading Pulmonology DataLoaders (Batch Size: {batch_size})...")
    train_loader, val_loader, test_loader = get_pulmonology_loaders(batch_size=batch_size)
    Logger.success(f"Loaded: {len(train_loader)} train batches ({len(train_loader.dataset)} samples), "
                   f"{len(val_loader)} val batches ({len(val_loader.dataset)} samples)")

    # 3. Setup Criterion & Optimizer
    # Class weights for medical class imbalance
    # Total: Normal=10192, COVID=3616, Lung_Opacity=6012, Viral_Pneumonia=1345
    class_weights = torch.tensor([1.0, 2.8, 1.7, 7.5]).to(device)
    if loss_type == "focal":
        criterion = get_loss_criterion("focal", class_weights=class_weights, gamma=2.0)
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 4. Create Experiment Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("core/models/pulmonology/experiments") / f"{model_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    Logger.success(f"Experiment Directory Created: {exp_dir}")

    # Save run metadata
    run_meta = {
        "model_name": model_name,
        "parameters": params,
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "loss_type": loss_type,
        "device": device,
        "start_time": datetime.now().isoformat()
    }
    with open(exp_dir / "experiment_meta.json", "w") as f:
        json.dump(run_meta, f, indent=4)

    # 5. Initialize Universal Trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_dir=str(exp_dir),
        patience=patience,
        accumulation_steps=1
    )

    # 6. Fit Model
    trainer.fit(num_epochs=epochs, resume=False)

    best_val_loss = trainer.best_val_loss
    best_val_acc = max(trainer.history['val_acc']) if trainer.history['val_acc'] else 0.0
    best_val_f1 = max(trainer.history['val_f1']) if trainer.history['val_f1'] else 0.0

    print("\n" + "=" * 60)
    print(f"{Colors.GREEN}{Colors.BOLD}✅ Finished Training: {model_name}{Colors.RESET}")
    print(f"  ↳ Best Validation Loss: {best_val_loss:.4f}")
    print(f"  ↳ Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  ↳ Best Validation F1-Score: {best_val_f1:.4f}")
    print(f"  ↳ Checkpoints & Curves: {exp_dir}")
    print("=" * 60 + "\n")

    return {
        "model_name": model_name,
        "exp_dir": str(exp_dir),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "params": params
    }

def main():
    parser = argparse.ArgumentParser(description="Pulmonology S.O.T.A. Model Benchmark Runner")
    parser.add_argument("--model", type=str, default="all", choices=SUPPORTED_MODELS + ["all"],
                        help="Model to train: DenseNet121, EfficientNetV2_S, ResNet50, ConvNeXt_Tiny, or all")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce", "weighted_ce"])
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    models_to_train = SUPPORTED_MODELS if args.model == "all" else [args.model]

    print("\n" + "#" * 70)
    print(f"{Colors.BOLD}{Colors.CYAN}🎯 Starting PanOrganNet Pulmonology Benchmark Suite{Colors.RESET}")
    print(f"   Candidate Models: {', '.join(models_to_train)}")
    print(f"   Epochs: {args.epochs} | Loss: {args.loss} | Batch Size: {args.batch_size}")
    print("#" * 70)

    summary_results = []
    for model_name in models_to_train:
        res = train_single_model(
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_type=args.loss,
            patience=args.patience
        )
        summary_results.append(res)

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN}🏆 BENCHMARK SUMMARY RESULTS{Colors.RESET}")
    print("=" * 70)
    print(f"{'Model':<18} | {'Params (M)':<10} | {'Best Val Acc':<14} | {'Best Val F1':<12} | {'Exp Directory'}")
    print("-" * 70)
    for r in summary_results:
        print(f"{r['model_name']:<18} | {r['params']['total_params_million']:<10} | "
              f"{r['best_val_acc']*100:>6.2f}%       | {r['best_val_f1']:>6.4f}       | {r['exp_dir']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
