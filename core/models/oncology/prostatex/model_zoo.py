import torch
import torch.nn as nn
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights
)

SUPPORTED_MODELS = ["EfficientNetV2_S", "DenseNet121", "ResNet50"]

def get_prostatex_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory method to retrieve configured architectures for PROSTATEx MRI analysis.
    """
    if model_name == "DenseNet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "EfficientNetV2_S":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported: {SUPPORTED_MODELS}")

    return model
