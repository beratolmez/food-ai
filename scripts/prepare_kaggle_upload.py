import os
import json
import argparse
from pathlib import Path

# Proje kök dizinini belirle
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_dataset_metadata(title: str, id_name: str, path: Path):
    """Kaggle Datasets API'si için meta veri dosyası oluşturur."""
    metadata = {
        "title": title,
        "id": f"<KAGGLE_KULLANICI_ADINIZ>/{id_name}",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    
    metadata_path = path / "dataset-metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata dosyası oluşturuldu: {metadata_path}")

def main():
    print("="*60)
    print("Kaggle Upload Hazırlığı")
    print("="*60)
    
    # 1. YOLO Data Metadata
    yolo_dir = PROJECT_ROOT / "data" / "detection" / "yolo"
    if yolo_dir.exists():
        create_dataset_metadata(
            title="HealfoAI YOLO Food Detection Data",
            id_name="healfoai-yolo-data",
            path=yolo_dir
        )
    else:
        print(f"❌ Klasör bulunamadı: {yolo_dir}")

    # 2. Classifier Data Metadata
    cls_dir = PROJECT_ROOT / "data" / "splits" / "classification"
    if cls_dir.exists():
        create_dataset_metadata(
            title="HealfoAI Food101 Custom Splits",
            id_name="healfoai-classifier-data",
            path=cls_dir
        )
    else:
        print(f"❌ Klasör bulunamadı: {cls_dir}")

    print("\nLütfen dataset-metadata.json dosyalarındaki <KAGGLE_KULLANICI_ADINIZ> kısmını kendi Kaggle kullanıcı adınız ile değiştirin!")

if __name__ == "__main__":
    main()
