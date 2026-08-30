"""
Model definitions for Cats vs Dogs binary classification.

Models:
  - SimpleCNN        : lightweight baseline, trains from scratch (~40 min on CPU, ~10 min GPU)
  - MobileNetV3Small : pretrained torchvision backbone, fine-tuned head only (~5 min/epoch GPU)

Factory:
  get_model(name, num_classes) -> nn.Module
"""
import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Baseline CNN (unchanged from main)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# MobileNetV3-Small (pretrained ImageNet backbone, fine-tuned head)
# ---------------------------------------------------------------------------
class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small with a custom 2-class classifier head.

    Strategy (fast convergence on limited data):
      - backbone: frozen by default (set freeze_backbone=False to unfreeze)
      - classifier head: replaced + trained from scratch

    For edge deployment: model is exported as CPU state_dict, so CUDA-trained
    weights load fine on CPU with `map_location='cpu'`.
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        # Load pretrained backbone
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Freeze the feature extractor
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Keep the feature extractor as-is
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Replace classifier: original has 576->1024->num_classes
        # We keep 576->256->num_classes (lighter for 2 classes)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, num_classes),
        )

        # Initialize new head weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def unfreeze_backbone(self):
        """Unfreeze all feature extractor layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "MobileNetV3Small": MobileNetV3Small,
}


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name: One of 'SimpleCNN', 'MobileNetV3Small'
        num_classes: Number of output classes (default 2)
        **kwargs: Additional kwargs forwarded to the model constructor
                  (e.g. freeze_backbone=False, dropout=0.3)
    Returns:
        nn.Module instance (not yet moved to device)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
