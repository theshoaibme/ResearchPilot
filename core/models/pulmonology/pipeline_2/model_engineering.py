import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ==============================================================================
# PIPELINE 2: EFFICIENTNET-V2-S MODEL ARCHITECTURE SETUP
# ==============================================================================
def get_efficientnet_v2_s_model(num_classes: int = 4, pretrained: bool = True):
    """
    Initializes an EfficientNetV2-S model for 4-class Chest X-ray classification.
    
    Why EfficientNetV2-S?
    - Uses Fused-MBConv layers for faster training and better parameter efficiency.
    - Progressive learning and neural architecture search (NAS) optimized for speed and accuracy.
    - Excellent feature representation for complex medical image pathologies with low GPU memory footprint.
    
    Args:
        num_classes: 4 classes (Normal, COVID, Lung_Opacity, Viral Pneumonia).
        pretrained: If True, loads ImageNet pretrained weights for transfer learning.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    
    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model
