import torch.nn as nn
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights
)

SUPPORTED_MODELS = ["EfficientNetV2_S", "DenseNet121", "ResNet50"]

def get_busi_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Initializes a transfer-learning model tailored for 3-class Breast Ultrasound diagnosis:
    - Normal (0)
    - Benign (1)
    - Malignant (2)
    """
    if model_name == "EfficientNetV2_S":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "DenseNet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
