import os
import torch
from data_engineering import get_dataloaders
from model_engineering import get_densenet_model, TrainingEngine

# ==============================================================================
# MAIN TRAINING PIPELINE ORCHESTRATOR
# ==============================================================================
# This script is the "Glue" that connects Data Engineering to Model Engineering.
# It acts as the command center for training our DenseNet121 MVP.

def main():
    # 1. Configuration
    # We are now inside core/models/pulmonology/pipeline_1
    manifest_path = '../../../../dataset/metadata/pulmonology/manifest.json'
    batch_size = 32
    num_epochs = 10
    
    # We want to train on a GPU if it's available (CUDA or Apple Silicon MPS). 
    # Otherwise, it falls back to CPU (which will be very slow).
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps' # Apple Silicon (M1/M2/M3)
    else:
        device = 'cpu'
        
    print(f"🚀 Starting Pipeline 01 (MVP) on device: {device.upper()}")
    
    # 2. Initialize Data Generators
    print("📦 Loading Data Generators (DataLoaders)...")
    train_loader, val_loader = get_dataloaders(manifest_path, batch_size=batch_size)
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    
    # 3. Initialize the Model
    print("🧠 Initializing DenseNet121...")
    # num_classes=4 because we have Normal, COVID, Lung_Opacity, Viral Pneumonia
    model = get_densenet_model(num_classes=4, pretrained=True)
    
    # 4. Initialize the Training Engine
    engine = TrainingEngine(model, train_loader, val_loader, device=device)
    
    # 5. The Training Loop (Epochs)
    best_val_acc = 0.0
    save_dir = '../../../../models/pulmonology' # Save the .pt weights at the root models directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🔥 Starting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Train for one epoch
        train_loss, train_acc = engine.train_epoch(epoch)
        print(f"📈 Train - Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
        
        # Validate for one epoch
        val_loss, val_acc = engine.validate_epoch()
        print(f"📊 Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
        
        # 6. Model Checkpointing (Save the best model)
        if val_acc > best_val_acc:
            print(f"⭐ Validation Accuracy improved from {best_val_acc*100:.2f}% to {val_acc*100:.2f}%. Saving model...")
            best_val_acc = val_acc
            
            # We save the model's "state_dict", which is a dictionary containing all the learned weights.
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_densenet121.pt'))
            
    print("\n🎉 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"💾 Model saved to: {os.path.join(save_dir, 'best_densenet121.pt')}")

if __name__ == '__main__':
    main()
