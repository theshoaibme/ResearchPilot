import os
import sys
import torch

# Add root project directory to path so we can import 'core' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from core.models.pulmonology.pipeline_2.data_engineering import get_dataloaders
from core.models.pulmonology.pipeline_2.model_engineering import get_efficientnet_v2_s_model
from core.models.pulmonology.pipeline_2.loss_engineering import get_loss_criterion
from core.engine.base_trainer import BaseTrainer, create_experiment_dir, Logger

# ==============================================================================
# PIPELINE 2 TRAINING ORCHESTRATOR (EFFICIENTNET-V2-S + FOCAL LOSS + V2 AUGMENTATIONS)
# ==============================================================================

def main():
    # 1. Configuration
    splits_dir = 'dataset/splits/pulmonology/chest_xray'
    images_dir = 'dataset/processed/pulmonology/chest_xray/images'
    batch_size = 32
    num_epochs = 10
    
    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    Logger.info(f"🚀 Starting Pipeline 02 (EfficientNetV2-S) on device: {device.upper()}")
    
    # 2. Initialize Data Generators & Class Weights
    Logger.info("📦 Loading Data Generators & Class Weights...")
    train_loader, val_loader, class_weights = get_dataloaders(splits_dir, images_dir, batch_size=batch_size)
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Class Weights: {class_weights.tolist()}")
    
    # 3. Initialize Model, Loss Function (Focal Loss), Optimizer, and Scheduler
    Logger.info("🧠 Initializing EfficientNetV2-S Model...")
    model = get_efficientnet_v2_s_model(num_classes=4, pretrained=True)
    
    criterion = get_loss_criterion(loss_type='focal', class_weights=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Advanced Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 4. Generate dynamic experiment folder for Version 2
    exp_dir = create_experiment_dir(module_name="pulmonology", model_name="EfficientNetV2_S")
    
    # 5. Initialize Universal Base Trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_dir=exp_dir,
        patience=5,
        accumulation_steps=1
    )
    
    # 6. Ignite Training (Version 2)
    trainer.fit(num_epochs=num_epochs, resume=False)

if __name__ == '__main__':
    main()
