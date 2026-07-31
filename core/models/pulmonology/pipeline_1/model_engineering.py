import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

# ==============================================================================
# 1. MODEL ARCHITECTURE SETUP
# ==============================================================================
def get_densenet_model(num_classes: int = 4, pretrained: bool = True):
    """
    Initializes a DenseNet121 model.
    
    Why DenseNet121?
    - "Dense" connections mean every layer is connected to every other layer.
    - This is excellent for Medical X-Rays because it preserves low-level features
      (like subtle bone structures or faint opacities) all the way to the final classification layer.
    - It requires fewer parameters than ResNet, saving GPU memory.
    
    Args:
        num_classes: We have 4 classes (Normal, COVID, Lung_Opacity, Viral Pneumonia).
        pretrained: If True, uses weights trained on ImageNet (Transfer Learning).
                    Transfer learning is crucial when medical datasets are small.
    """
    # 1. Load the architecture and optional pre-trained weights
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    
    # 2. Replace the final classification head (the fully connected layer)
    # DenseNet originally predicts 1000 classes (ImageNet). We change it to output 4 classes.
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model

# ==============================================================================
# 2. TRAINING ENGINE (THE LOOP)
# ==============================================================================
class TrainingEngine:
    """
    The Training Engine abstracts away the complex PyTorch training loop.
    It handles Forward Passes, Loss Calculation, Backpropagation, and Validation.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss Function: CrossEntropyLoss is standard for Multi-Class Classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: Adam is great for fast convergence. Learning rate (lr) is set to a standard 1e-4.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train_epoch(self, epoch_idx):
        """
        Executes one full pass through the Training dataset.
        """
        self.model.train() # Set model to training mode (enables Dropout/BatchNorm updates)
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to the active device (CPU or GPU)
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 1. Zero the gradients (PyTorch accumulates gradients by default, we must clear them)
            self.optimizer.zero_grad()
            
            # 2. Forward Pass: Pass images through the model to get predictions
            outputs = self.model(images)
            
            # 3. Calculate Loss: Compare predictions (outputs) against the ground truth (labels)
            loss = self.criterion(outputs, labels)
            
            # 4. Backward Pass: Compute gradients (how much each weight should change)
            loss.backward()
            
            # 5. Optimization Step: Update the model's weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch_idx}] Batch [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    @torch.no_grad() # Disable gradient tracking for validation (saves memory and speeds up)
    def validate_epoch(self):
        """
        Executes one full pass through the Validation dataset.
        No learning happens here. We just evaluate how well the model generalizes.
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy
