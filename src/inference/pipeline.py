"""End-to-end food recognition pipeline.

Chains YOLO detection → crop → EfficientNet classification.

Usage:
    pipeline = FoodPipeline(
        detector_path="exports/detector/best.pt",
        classifier_path="exports/classifier/best_model.pt",
        class_mapping_path="exports/classifier/class_mapping.json",
    )
    results = pipeline.run(image)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src.inference.detector import FoodDetector
from src.inference.classifier import FoodClassifier


class FoodPipeline:
    """Detect food regions then classify each one.

    This is the main entry point for the backend API.
    """

    def __init__(
        self,
        detector_path: Union[str, Path],
        classifier_path: Union[str, Path],
        class_mapping_path: Union[str, Path],
        det_conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        """
        Args:
            detector_path: Path to YOLO best.pt.
            classifier_path: Path to EfficientNet best_model.pt.
            class_mapping_path: Path to class_mapping.json.
            det_conf_threshold: YOLO confidence threshold.
            device: 'cpu' or 'cuda'.
        """
        self.detector = FoodDetector(
            model_path=detector_path,
            conf_threshold=det_conf_threshold,
            device=device,
        )
        self.classifier = FoodClassifier(
            model_path=classifier_path,
            class_mapping_path=class_mapping_path,
            device=device,
        )

    def run(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        top_k: int = 3,
    ) -> dict:
        """Run the full pipeline on an image.

        Args:
            image: Input image (PIL, numpy, or path).
            top_k: Number of top class predictions per detection.

        Returns:
            Dict with keys:
                num_detections (int): Number of food items found.
                items (list[dict]): Per-detection results, each containing:
                    - bbox (list[int]): [x1, y1, x2, y2]
                    - det_confidence (float): YOLO detection confidence.
                    - label (str): Top-1 predicted food class.
                    - confidence (float): Classifier confidence.
                    - top_k (list[dict]): Top-K classifier predictions.
        """
        crops, boxes = self.detector.detect(image)

        items = []
        for crop, box in zip(crops, boxes):
            cls_result = self.classifier.predict(crop, top_k=top_k)
            items.append({
                "bbox": box["xyxy"],
                "det_confidence": box["conf"],
                "label": cls_result["label"],
                "confidence": cls_result["confidence"],
                "top_k": cls_result["top_k"],
            })

        return {
            "num_detections": len(items),
            "items": items,
        }

    def run_bytes(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """Run pipeline from raw image bytes (for HTTP/API use).

        Args:
            image_bytes: Raw image bytes (e.g. from file upload).
            top_k: Number of top class predictions per detection.

        Returns:
            Same structure as `run()`.
        """
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.run(image, top_k=top_k)
