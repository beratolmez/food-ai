from __future__ import annotations

stats[f"mapped_{split_name}"] += 1


def summarize_split(root: Path) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    if not root.exists():
        return summary

    for class_dir in sorted(root.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            summary[class_dir.name] = count
    return summary


def main() -> None:
    classes_json = load_json(CLASSES_PATH)
    mapping = load_json(MAPPING_PATH)
    allowed_classes = build_allowed_classes(classes_json)
    validate_mapping(mapping, allowed_classes)

    images_root = FOOD101_ROOT / "images"
    meta_root = FOOD101_ROOT / "meta"
    train_file = meta_root / "train.txt"
    test_file = meta_root / "test.txt"

    if not images_root.exists():
        raise FileNotFoundError(
            f"Food-101 images folder not found: {images_root}""Expected structure: data/classification/raw/food101/images/..."
        )

    ensure_dir(PROCESSED_ROOT)
    ensure_dir(SPLITS_ROOT / "train")
    ensure_dir(SPLITS_ROOT / "test")

    train_entries = read_split_file(train_file)
    test_entries = read_split_file(test_file)

    stats: Dict[str, int] = {
        "copied_processed": 0,
        "copied_split": 0,
        "mapped_train": 0,
        "mapped_test": 0,
        "skipped_unmapped": 0,
        "missing_files": 0,
    }

    for entry in train_entries:
        copy_if_mapped(entry, images_root, mapping, "train", stats)

    for entry in test_entries:
        copy_if_mapped(entry, images_root, mapping, "test", stats)

    print("=== Preparation Complete ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("=== Train Split Summary ===")
    train_summary = summarize_split(SPLITS_ROOT / "train")
    for class_name, count in train_summary.items():
        print(f"{class_name}: {count}")

    print("=== Test Split Summary ===")
    test_summary = summarize_split(SPLITS_ROOT / "test")
    for class_name, count in test_summary.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()