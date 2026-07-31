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
