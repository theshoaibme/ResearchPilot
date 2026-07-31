import os
import sys
import torch
import torch.nn as nn

# Add root project directory to path so we can import 'core' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from core.models.pulmonology.pipeline_1.data_engineering import get_dataloaders
from core.models.pulmonology.pipeline_1.model_engineering import get_densenet_model
from core.engine.base_trainer import BaseTrainer, create_experiment_dir, Logger

# ==============================================================================
# MAIN TRAINING PIPELINE ORCHESTRATOR (USING S.O.T.A ENGINE)
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
        
    Logger.info(f"🚀 Starting Pipeline 01 (MVP) on device: {device.upper()}")
    
    # 2. Initialize Data Generators
    Logger.info("📦 Loading Data Generators (DataLoaders)...")
    train_loader, val_loader = get_dataloaders(splits_dir, images_dir, batch_size=batch_size)
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    
    # 3. Initialize the Model, Optimizer, and Loss
    Logger.info("🧠 Initializing DenseNet121...")
    model = get_densenet_model(num_classes=4, pretrained=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Adding a Learning Rate Scheduler! (Halves the LR if validation doesn't improve for 2 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Generate a dynamic timestamped folder for this specific run
    # Base dir starts at root, so we just use 'core/models'
    exp_dir = create_experiment_dir(module_name="pulmonology", model_name="DenseNet121")
    
    # 5. Initialize the Universal Base Trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler, # Pass the scheduler!
        device=device,
        exp_dir=exp_dir,
        patience=5,          # Early stopping patience
        accumulation_steps=1 # Set to 2 or 4 if running out of GPU memory
    )
    
    # 6. Ignite! (Use resume=True to survive Load-Shedding)
    trainer.fit(num_epochs=num_epochs, resume=False)

if __name__ == '__main__':
    main()
