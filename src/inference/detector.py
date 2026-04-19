"""Food Detection inference wrapper using YOLOv8.

Loads a trained YOLOv8 model and returns bounding box crops from an input image.

Usage:
    detector = FoodDetector("exports/detector/best.pt")
    crops, boxes = detector.detect(image)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class FoodDetector:
    """YOLO-based food detector that returns cropped food regions."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to trained best.pt YOLO weights.
            conf_threshold: Minimum confidence score for a detection.
            iou_threshold: IoU threshold for NMS.
            device: 'cpu' or 'cuda'.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(
        self, image: Union[Image.Image, np.ndarray, str, Path]
    ) -> tuple[list[Image.Image], list[dict]]:
        """Run detection on an image and return crops + metadata.

        Args:
            image: PIL Image, numpy array, or path to image file.

        Returns:
            crops: List of PIL Image crops (one per detected food item).
            boxes: List of dicts with keys: xyxy, conf, class_id, class_name.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        crops = []
        boxes = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Add small padding around the crop (3%)
                w, h = pil_image.size
                pad_x = int((x2 - x1) * 0.03)
                pad_y = int((y2 - y1) * 0.03)
                x1c = max(0, x1 - pad_x)
                y1c = max(0, y1 - pad_y)
                x2c = min(w, x2 + pad_x)
                y2c = min(h, y2 + pad_y)

                crop = pil_image.crop((x1c, y1c, x2c, y2c))
                crops.append(crop)
                boxes.append({
                    "xyxy": [x1, y1, x2, y2],
                    "conf": round(conf, 4),
                    "class_id": class_id,
                    "class_name": "food",
                })

        return crops, boxes

    def detect_raw(self, image: Union[Image.Image, np.ndarray, str, Path]):
        """Return raw Ultralytics results object (for advanced use)."""
        return self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
