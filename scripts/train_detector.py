"""Train YOLOv8n detector on the prepared UECFOOD256 "food" dataset.

This script should be run on a GPU environment (Colab/Kaggle).
It uses Ultralytics YOLOv8 library.

Usage:
    python scripts/train_detector.py
    python scripts/train_detector.py --epochs 100 --batch 32
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default to Kaggle working dir if inside Kaggle, else local runs
KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_RUNS_DIR = KAGGLE_WORKING / "runs" if KAGGLE_WORKING.exists() else PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 Detector")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "detector_data.yaml"),
        help="Path to detector_data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_RUNS_DIR / "detector"),
        help="Project directory to save runs",
    )
    parser.add_argument("--name", type=str, default="yolov8n_food_m1", help="Experiment name")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Base model to load (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Initiating YOLOv8 Food Detector Training")
    print("=" * 60)
    print(f"Data config: {args.data}")
    print(f"Base model : {args.model}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch}")
    print(f"Image size : {args.imgsz}")
    print(f"Output dir : {Path(args.project) / args.name}")
    print("=" * 60)

    # 1. Load the model
    # Load a pretrained model (recommended for training)
    model = YOLO(args.model)

    # 2. Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        device=0,  # Assumes 1 GPU is available. Use 'cpu' if no GPU.
        exist_ok=True,
    )

    print("\nTraining completed successfully!")
    print(f"Best model saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
