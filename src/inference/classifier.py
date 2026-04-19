"""Food Classification inference wrapper using EfficientNet-B0.

Loads trained PyTorch weights and classifies a cropped food image.

Usage:
    clf = FoodClassifier("exports/classifier/best_model.pt", "exports/classifier/class_mapping.json")
    label, confidence, top3 = clf.predict(crop_image)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class FoodClassifier:
    """EfficientNet-B0 food classifier with top-K predictions."""

    # ImageNet normalization (same as training)
    _NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        model_path: Union[str, Path],
        class_mapping_path: Union[str, Path],
        image_size: int = 224,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to best_model.pt checkpoint.
            class_mapping_path: Path to class_mapping.json (class_name → idx).
            image_size: Model input size (224 for EfficientNet-B0).
            device: 'cpu' or 'cuda'.
        """
        from torchvision import models

        self.device = torch.device(device)
        self.image_size = image_size

        # Load class mapping (name → idx), invert to idx → name
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            class_to_idx: dict[str, int] = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)

        # Build model architecture (must match training)
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        import torch.nn as nn
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Inference transform (deterministic — no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self._NORMALIZE,
        ])

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        top_k: int = 3,
    ) -> dict:
        """Classify a food image and return top-K predictions.

        Args:
            image: PIL Image, numpy array, or path.
            top_k: Number of top predictions to return.

        Returns:
            Dict with keys:
                label (str): Top-1 predicted class name.
                confidence (float): Top-1 confidence score (0-1).
                top_k (list[dict]): Top-K results, each with 'label' and 'confidence'.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_k_vals, top_k_idxs = torch.topk(probs, min(top_k, self.num_classes))

        top_results = [
            {
                "label": self.idx_to_class[idx.item()],
                "confidence": round(val.item(), 4),
            }
            for val, idx in zip(top_k_vals, top_k_idxs)
        ]

        return {
            "label": top_results[0]["label"],
            "confidence": top_results[0]["confidence"],
            "top_k": top_results,
        }
