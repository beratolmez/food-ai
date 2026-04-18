"""Validate YOLO format dataset after conversion.

Checks:
- All images have corresponding label files
- Label format is valid (class_id cx cy cw ch)
- Bounding box values are within [0, 1]
- Images are readable
- Prints split statistics

Usage:
    python scripts/validate_yolo_data.py
    python scripts/validate_yolo_data.py --yolo-root data/detection/yolo
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_ROOT = PROJECT_ROOT / "data" / "detection" / "yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=Path(os.environ.get("HEALFOAI_YOLO_OUTPUT", str(DEFAULT_YOLO_ROOT))),
        help="YOLO dataset root directory",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        default=False,
        help="Verify images are readable (slower)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample labels to print per split",
    )
    return parser.parse_args()


def validate_label_file(label_path: Path) -> dict:
    """Validate a single YOLO label file.

    Returns:
        Dict with keys: valid (bool), num_boxes (int), errors (list[str])
    """
    errors = []
    num_boxes = 0

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: expected 5 values, got {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                errors.append(f"Line {line_num}: invalid numeric values")
                continue

            if class_id < 0:
                errors.append(f"Line {line_num}: negative class_id={class_id}")

            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0 or val > 1:
                    errors.append(f"Line {line_num}: {name}={val:.4f} outside [0, 1]")

            if w <= 0 or h <= 0:
                errors.append(f"Line {line_num}: degenerate box w={w:.4f} h={h:.4f}")

            num_boxes += 1

    return {"valid": len(errors) == 0, "num_boxes": num_boxes, "errors": errors}


def validate_split(
    yolo_root: Path, split_name: str, check_images: bool, show_samples: int
) -> dict:
    """Validate a single split (train/val/test).

    Returns:
        Dict with statistics.
    """
    img_dir = yolo_root / "images" / split_name
    lbl_dir = yolo_root / "labels" / split_name

    if not img_dir.exists():
        return {"exists": False, "error": f"Image dir not found: {img_dir}"}
    if not lbl_dir.exists():
        return {"exists": False, "error": f"Label dir not found: {lbl_dir}"}

    image_files = sorted(img_dir.glob("*.jpg"))
    label_files = sorted(lbl_dir.glob("*.txt"))

    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    total_boxes = 0
    invalid_labels = 0
    all_errors = []
    unreadable_images = []

    for lbl_file in tqdm(label_files, desc=f"Validating {split_name} labels", unit="file"):
        result = validate_label_file(lbl_file)
        total_boxes += result["num_boxes"]
        if not result["valid"]:
            invalid_labels += 1
            all_errors.extend([f"{lbl_file.name}: {e}" for e in result["errors"]])

    if check_images:
        for img_file in tqdm(image_files, desc=f"Checking {split_name} images", unit="img"):
            try:
                with Image.open(img_file) as img:
                    img.verify()
            except Exception as e:
                unreadable_images.append(f"{img_file.name}: {e}")

    # Print per-split report
    print(f"\n{'-' * 40}")
    print(f"Split: {split_name}")
    print(f"{'-' * 40}")
    print(f"  Images: {len(image_files)}")
    print(f"  Labels: {len(label_files)}")
    print(f"  Total bboxes: {total_boxes}")
    print(f"  Avg bboxes/image: {total_boxes / max(len(label_files), 1):.2f}")

    if missing_labels:
        print(f"  [WARN] Missing labels: {len(missing_labels)}")
    if orphan_labels:
        print(f"  [WARN] Orphan labels (no image): {len(orphan_labels)}")
    if invalid_labels:
        print(f"  [FAIL] Invalid label files: {invalid_labels}")
        for err in all_errors[:5]:
            print(f"     -> {err}")
    if unreadable_images:
        print(f"  [FAIL] Unreadable images: {len(unreadable_images)}")

    if not missing_labels and not orphan_labels and not invalid_labels and not unreadable_images:
        print(f"  [OK] All checks passed!")

    # Show sample labels
    if show_samples > 0 and label_files:
        print(f"\n  Sample labels (first {show_samples}):")
        for lbl_file in label_files[:show_samples]:
            with open(lbl_file, "r") as f:
                content = f.read().strip()
            print(f"    {lbl_file.name}: {content[:80]}")

    return {
        "exists": True,
        "num_images": len(image_files),
        "num_labels": len(label_files),
        "total_boxes": total_boxes,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "invalid_labels": invalid_labels,
        "unreadable_images": len(unreadable_images),
    }


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("YOLO Dataset Validation")
    print("=" * 50)
    print(f"Root: {args.yolo_root}")

    if not args.yolo_root.exists():
        print(f"[FAIL] YOLO root directory not found: {args.yolo_root}")
        print("Run prepare_uec_for_yolo.py first!")
        return

    results = {}
    for split in ["train", "val", "test"]:
        results[split] = validate_split(args.yolo_root, split, args.check_images, args.show_samples)

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")

    total_images = 0
    total_boxes = 0
    all_valid = True

    for split, result in results.items():
        if not result.get("exists", False):
            print(f"  {split}: NOT FOUND")
            all_valid = False
            continue

        total_images += result["num_images"]
        total_boxes += result["total_boxes"]
        issues = result["missing_labels"] + result["orphan_labels"] + result["invalid_labels"]
        status = "[OK]" if issues == 0 else "[WARN]"
        print(f"  {status} {split}: {result['num_images']} images, {result['total_boxes']} boxes")

        if issues > 0:
            all_valid = False

    print(f"\n  Total: {total_images} images, {total_boxes} boxes")
    print(f"\n  Overall: {'[OK] ALL VALID' if all_valid else '[WARN] ISSUES FOUND'}")


if __name__ == "__main__":
    main()
