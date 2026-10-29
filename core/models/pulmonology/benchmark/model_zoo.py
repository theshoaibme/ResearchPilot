import torch
import torch.nn as nn
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

SUPPORTED_MODELS = ["DenseNet121", "EfficientNetV2_S", "ResNet50", "ConvNeXt_Tiny"]

def get_model(model_name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Initializes a medical classification model with pre-trained weights
    and a custom classification head for 4-class Chest X-ray diagnosis.

    Supported architectures:
      - DenseNet121: S.O.T.A. medical imaging baseline with dense connectivity.
      - EfficientNetV2_S: High-accuracy Fused-MBConv CNN with low parameter footprint.
      - ResNet50: Classic deep residual network benchmark.
      - ConvNeXt_Tiny: Modernized pure convolutional architecture.
    """
    if model_name == "DenseNet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "EfficientNetV2_S":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "ConvNeXt_Tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {SUPPORTED_MODELS}")

    return model

def count_parameters(model: nn.Module) -> dict:
    """Returns total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_params_million": round(total_params / 1e6, 2)
    }
