"""Model definitions for Food Classification."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def create_classifier(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = True
) -> nn.Module:
    """Create a food classification model.
    
    Args:
        model_name: Name of the model architecture ('efficientnet_b0', 'mobilenet_v3_large')
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet weights
        
    Returns:
        PyTorch nn.Module
    """
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        # Replace the classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        # Replace the classifier head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'efficientnet_b0' or 'mobilenet_v3_large'.")
        
    return model


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all layers except the classifier head for Stage 1 training."""
    for param in model.parameters():
        param.requires_grad = False
        
    if "efficientnet" in model_name:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif "mobilenet" in model_name:
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers for Stage 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
