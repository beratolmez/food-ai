"""Prepare Food-101 dataset for classification training.

Maps Food-101 original classes → hedef sınıflar (classes_v2.json) using
label_mapping_v2.json. Creates train/val/test splits with proper directory
structure for PyTorch ImageFolder.

Usage:
    python scripts/prepare_food101_classifier.py
    python scripts/prepare_food101_classifier.py --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Food-101 for classification")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("HEALFOAI_DATA_ROOT", str(DEFAULT_DATA_ROOT))),
        help="Root directory containing raw datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "classification",
        help="Output root for train/val/test splits",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio from train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "classes_v2.json",
        help="Path to classes config",
    )
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "label_mapping_v2.json",
        help="Path to label mapping config",
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        default=False,
        help="Use symlinks instead of copying (saves disk space)",
    )
    return parser.parse_args()


def resolve_food101_root(data_root: Path) -> Path:
    """Find Food-101 root directory containing 'images' and 'meta' folders."""
    candidates = [
        data_root / "food-101" / "food-101",
        data_root / "food-101",
        data_root / "food101" / "food-101",
        data_root / "food101",
    ]

    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "meta").exists():
            return candidate

    raise FileNotFoundError(
        f"Food-101 root not found under {data_root}. "
        "Expected a folder containing both 'images/' and 'meta/' directories."
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_split_file(split_file: Path) -> list[str]:
    """Read Food-101 split file (train.txt or test.txt).

    Returns:
        List of 'class_name/image_id' strings.
    """
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    args = parse_args()

    # Load configs
    classes = load_json(args.classes_config)
    mapping_raw = load_json(args.mapping_config)

    # Extract food101 mapping
    food101_mapping = mapping_raw.get("food101", mapping_raw)
    valid_target_classes = set(classes.values())
    unmapped_policy = mapping_raw.get("_meta", {}).get("unmapped_policy", "other")

    print(f"Classes config: {len(classes)} classes")
    print(f"Food-101 mapping: {len(food101_mapping)} entries")
    print(f"Unmapped policy: {unmapped_policy}")

    # Resolve Food-101 root
    food101_root = resolve_food101_root(args.data_root)
    images_root = food101_root / "images"
    meta_root = food101_root / "meta"
    print(f"Food-101 root: {food101_root}")

    # Read original splits
    train_entries = read_split_file(meta_root / "train.txt")
    test_entries = read_split_file(meta_root / "test.txt")
    print(f"Original train: {len(train_entries)}, test: {len(test_entries)}")

    # Prepare output directories
    for split in ["train", "val", "test"]:
        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
        "skipped_unmapped": 0,
        "missing_files": 0,
    }

    def map_class(src_class: str) -> str | None:
        """Map a Food-101 class to target class."""
        if src_class in food101_mapping:
            target = food101_mapping[src_class]
            if target in valid_target_classes:
                return target
        if unmapped_policy == "other" and "other" in valid_target_classes:
            return "other"
        return None

    def process_entries(
        entries: list[str], split_name: str
    ) -> list[tuple[Path, str, str]]:
        """Process entries and return list of (src_path, target_class, filename)."""
        results = []
        for entry in entries:
            src_class, img_id = entry.split("/")
            target_class = map_class(src_class)

            if target_class is None:
                stats["skipped_unmapped"] += 1
                continue

            src_path = images_root / src_class / f"{img_id}.jpg"
            if not src_path.exists():
                stats["missing_files"] += 1
                continue

            # Use unique filename to avoid collisions from merged classes
            filename = f"{src_class}_{img_id}.jpg"
            results.append((src_path, target_class, filename))

        return results

    # Process test set (kept as-is)
    print("\nProcessing test split...")
    test_items = process_entries(test_entries, "test")
    for src_path, target_class, filename in tqdm(test_items, desc="Test", unit="img"):
        dst_dir = args.output_root / "test" / target_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / filename
        if not dst_path.exists():
            if args.use_symlinks:
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        stats["test"][target_class] += 1

    # Process train set → split into train + val
    print("Processing train split (with val split)...")
    train_items = process_entries(train_entries, "train")

    # Stratified split: shuffle within each class, then split
    random.seed(args.seed)
    class_groups: dict[str, list] = {}
    for item in train_items:
        target_class = item[1]
        if target_class not in class_groups:
            class_groups[target_class] = []
        class_groups[target_class].append(item)

    train_final = []
    val_final = []
    for cls, items in class_groups.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_ratio))
        val_final.extend(items[:n_val])
        train_final.extend(items[n_val:])

    # Write train
    for src_path, target_class, filename in tqdm(train_final, desc="Train", unit="img"):
        dst_dir = args.output_root / "train" / target_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / filename
        if not dst_path.exists():
            if args.use_symlinks:
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        stats["train"][target_class] += 1

    # Write val
    for src_path, target_class, filename in tqdm(val_final, desc="Val", unit="img"):
        dst_dir = args.output_root / "val" / target_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / filename
        if not dst_path.exists():
            if args.use_symlinks:
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        stats["val"][target_class] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Food-101 Classification Data Preparation Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_root}")
    print(f"Skipped (unmapped): {stats['skipped_unmapped']}")
    print(f"Missing files: {stats['missing_files']}")

    for split in ["train", "val", "test"]:
        counter = stats[split]
        total = sum(counter.values())
        print(f"\n{split.upper()} ({total} images, {len(counter)} classes):")
        for cls in sorted(counter.keys()):
            print(f"  {cls:25s}: {counter[cls]:5d}")

    # Save metadata
    meta = {
        "source": "food-101",
        "classes_config": str(args.classes_config),
        "mapping_config": str(args.mapping_config),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "splits": {
            split: {"total": sum(stats[split].values()), "per_class": dict(stats[split])}
            for split in ["train", "val", "test"]
        },
    }
    meta_path = args.output_root / "preparation_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {meta_path}")


if __name__ == "__main__":
    main()