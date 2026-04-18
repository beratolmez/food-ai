"""Convert UECFOOD256 dataset to YOLO format for single-class food detection.

This script reads UECFOOD256 bounding box annotations (bb_info.txt) from each
category folder, converts them to YOLO format (normalized cx, cy, w, h), and
splits the data into train/val/test sets.

All 256 food categories are mapped to a single class: 0 (food).

Usage:
    python scripts/prepare_uec_for_yolo.py
    python scripts/prepare_uec_for_yolo.py --val-ratio 0.15 --test-ratio 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UEC_ROOT = PROJECT_ROOT / "data" / "raw" / "UECFOOD256"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "detection" / "yolo"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert UECFOOD256 to YOLO single-class format"
    )
    parser.add_argument(
        "--uec-root",
        type=Path,
        default=Path(os.environ.get("HEALFOAI_UEC_ROOT", str(DEFAULT_UEC_ROOT))),
        help="Path to UECFOOD256 root directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("HEALFOAI_YOLO_OUTPUT", str(DEFAULT_OUTPUT_ROOT))),
        help="Output directory for YOLO format data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def read_bb_info(bb_file: Path) -> dict[str, list[list[int]]]:
    """Read bounding box info from a bb_info.txt file.

    Args:
        bb_file: Path to bb_info.txt

    Returns:
        Dict mapping image_name (without extension) → list of [x1, y1, x2, y2] bboxes.
        Multiple bboxes per image are supported.
    """
    bboxes: dict[str, list[list[int]]] = {}

    with open(bb_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("img"):
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            img_id, x1, y1, x2, y2 = parts
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            if img_id not in bboxes:
                bboxes[img_id] = []
            bboxes[img_id].append(bbox)

    return bboxes


def convert_bbox_to_yolo(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    """Convert pixel bbox (x1,y1,x2,y2) to YOLO format (cx, cy, w, h) normalized.

    Args:
        x1, y1, x2, y2: Pixel coordinates of the bounding box.
        img_w, img_h: Image dimensions.

    Returns:
        Tuple of (cx, cy, w, h) normalized to [0, 1], or None if invalid.
    """
    # Clamp to image bounds
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    box_w = x2 - x1
    box_h = y2 - y1

    # Skip degenerate boxes
    if box_w <= 2 or box_h <= 2:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = box_w / img_w
    h = box_h / img_h

    return (cx, cy, w, h)


def process_category(
    cat_dir: Path,
) -> list[dict]:
    """Process a single UECFOOD256 category folder.

    Args:
        cat_dir: Path to category directory (e.g., data/raw/UECFOOD256/1/)

    Returns:
        List of dicts with keys: image_path, labels (list of YOLO format strings)
    """
    bb_file = cat_dir / "bb_info.txt"
    if not bb_file.exists():
        return []

    bboxes = read_bb_info(bb_file)
    samples = []

    for img_id, bbox_list in bboxes.items():
        # Try to find the image file
        img_path = cat_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        yolo_labels = []
        for bbox in bbox_list:
            result = convert_bbox_to_yolo(*bbox, img_w, img_h)
            if result is not None:
                cx, cy, w, h = result
                # Class 0 = food (single class)
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_labels:
            samples.append({
                "image_path": img_path,
                "labels": yolo_labels,
                "img_id": f"cat{cat_dir.name}_{img_id}",
            })

    return samples


def split_data(
    samples: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Split samples into train/val/test sets.

    Args:
        samples: List of sample dicts.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        Dict with keys 'train', 'val', 'test' → lists of samples.
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)

    splits = {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }
    return splits


def write_yolo_data(
    splits: dict[str, list[dict]],
    output_root: Path,
) -> dict[str, int]:
    """Write YOLO format images and labels to output directory.

    Args:
        splits: Dict of split_name → list of samples.
        output_root: Root output directory.

    Returns:
        Dict with split counts.
    """
    stats = {}

    for split_name, samples in splits.items():
        img_dir = output_root / "images" / split_name
        lbl_dir = output_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=f"Writing {split_name}", unit="img"):
            img_id = sample["img_id"]
            src_img = sample["image_path"]

            dst_img = img_dir / f"{img_id}.jpg"
            dst_lbl = lbl_dir / f"{img_id}.txt"

            # Copy image
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Write label
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(sample["labels"]) + "\n")

        stats[split_name] = len(samples)

    return stats


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print(f"UECFOOD256 root: {args.uec_root}")
    print(f"Output root: {args.output_root}")
    print(f"Split ratios — val: {args.val_ratio}, test: {args.test_ratio}")
    print()

    if not args.uec_root.exists():
        raise FileNotFoundError(f"UECFOOD256 root not found: {args.uec_root}")

    # Collect all samples from all category folders
    all_samples = []
    category_dirs = sorted(
        [d for d in args.uec_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    print(f"Found {len(category_dirs)} category directories")

    for cat_dir in tqdm(category_dirs, desc="Processing categories", unit="cat"):
        samples = process_category(cat_dir)
        all_samples.extend(samples)

    print(f"\nTotal samples with valid bboxes: {len(all_samples)}")

    if not all_samples:
        raise RuntimeError("No valid samples found! Check UECFOOD256 data path.")

    # Split data
    splits = split_data(all_samples, args.val_ratio, args.test_ratio, args.seed)

    # Write YOLO format
    stats = write_yolo_data(splits, args.output_root)

    # Print summary
    print("\n" + "=" * 50)
    print("UECFOOD256 -> YOLO Conversion Complete!")
    print("=" * 50)
    print(f"Output directory: {args.output_root}")
    for split_name, count in stats.items():
        print(f"  {split_name}: {count} images")
    print(f"  Total: {sum(stats.values())} images")
    print(f"  Classes: 1 (food)")

    # Save conversion metadata
    meta = {
        "source": "UECFOOD256",
        "num_classes": 1,
        "class_names": ["food"],
        "splits": stats,
        "total_samples": sum(stats.values()),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    meta_path = args.output_root / "conversion_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
