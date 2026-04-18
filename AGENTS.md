# AGENTS.md — HealfoAI ML Proje Kuralları

Bu dosya, AI asistanların bu projede çalışırken uyması gereken kuralları ve convention'ları tanımlar.

## 🎯 Proje Özeti

HealfoAI-ML, mobil kalori takip uygulaması için besin tanıma pipeline'ı geliştiren bir ML projesidir.
- **Detection:** YOLOv8n ile tek sınıf ("food") detection
- **Classification:** EfficientNet-B0 / MobileNetV3 ile 25-50 sınıf + "other" classification
- **Framework:** PyTorch (tek framework)
- **API:** FastAPI (local-first → cloud)

## 📁 Proje Yapısı Kuralları

```
configs/   → Sınıf listeleri, mapping'ler, training config'leri (JSON/YAML)
data/      → Ham ve işlenmiş veri (gitignore'da, repo'ya dahil değil)
src/       → Ana kaynak kodu (modüller, inference pipeline)
scripts/   → Çalıştırılabilir scriptler (train, evaluate, export, prepare)
api/       → FastAPI backend
notebooks/ → EDA ve deneme notebook'ları
runs/      → Training çıktıları (gitignore'da)
exports/   → Export edilmiş modeller (gitignore'da)
evaluation/→ Değerlendirme raporları ve görselleri
```

## 🐍 Python Kuralları

### Genel
- Python 3.10+
- Type hint'ler kullan (fonksiyon signature'larında zorunlu)
- Docstring: Google style
- Line length: 100 karakter (soft limit)
- Import sırası: stdlib → third-party → local (`isort` uyumlu)

### Naming Convention
- Dosya isimleri: `snake_case.py`
- Sınıf isimleri: `PascalCase`
- Fonksiyon/değişken isimleri: `snake_case`
- Sabitler: `UPPER_SNAKE_CASE`
- Config dosyaları: `kebab-case` veya `snake_case`

### Path Yönetimi
- Tüm path'ler `pathlib.Path` ile yönetilmeli
- Hardcoded absolute path YASAK
- Proje root'u: `Path(__file__).resolve().parents[N]` ile bulunmalı
- Data path'leri `configs/` altındaki config dosyalarından veya environment variable'lardan okunmalı

### Error Handling
- Custom exception class'ları kullan
- Script'lerde `try/except` ile anlamlı hata mesajları ver
- Asla sessizce hata yutma (`except: pass` YASAK)

## 🤖 Model Kuralları

### Detection (YOLO)
- Framework: Ultralytics YOLOv8
- Model: `yolov8n.pt` (nano)
- Tek sınıf: `food` (class_id=0)
- Input size: 640x640
- Confidence threshold: 0.25 (default, konfigüre edilebilir)
- NMS IoU threshold: 0.45

### Classification
- Framework: PyTorch (torchvision)
- Backbone: EfficientNet-B0 veya MobileNetV3-Large (ImageNet pretrained)
- Input size: 224x224
- Normalization: ImageNet mean/std
- Output: num_classes = len(classes) + 1 ("other")

### Training Convention
- Tüm training script'leri `scripts/` altında
- Config dosyaları `configs/` altında YAML formatında
- Çıktılar `runs/{detector|classifier}/{experiment_name}/` altında
- Best model: `best.pt` veya `best.pth`
- Training log: CSV formatında

### Export Convention
- Export edilen modeller `exports/{detector|classifier}/` altında
- Her export'un yanında metadata JSON'ı olmalı (input_size, class_names, version)
- Format: ONNX (backend) + TFLite (mobile, opsiyonel)

## 📊 Veri Kuralları

### Dataset Yapısı
- Ham veri: `data/raw/` (dokunulmaz, read-only)
- İşlenmiş veri: `data/processed/`
- Split'ler: `data/splits/` (train/val/test)
- YOLO formatı: `data/detection/yolo/{images|labels}/{train|val|test}/`

### Augmentation
- Training: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation
- Validation/Test: Resize + CenterCrop (deterministic)
- MixUp/CutMix: opsiyonel, dikkatli kullan

## 🔧 Config Dosyaları

### classes_v2.json
```json
{
  "0": "pizza",
  "1": "burger",
  ...
  "N": "other"
}
```

### label_mapping
- Food-101 orijinal sınıf → hedef sınıf mapping'i
- Birden fazla kaynak sınıf aynı hedefe map olabilir
- Map edilmeyen sınıflar "other"a gider (eğer kullanılacaksa)

## 🔌 API Kuralları

### Endpoint Convention
- `POST /predict` → görüntü yükle, tahmin al
- `GET /health` → sağlık kontrolü
- `GET /classes` → desteklenen sınıflar
- `GET /model-info` → model versiyonu ve metadata

### Response Format
- Tüm response'lar JSON
- Hata durumunda HTTP status code + detail mesajı
- Prediction response'ta her zaman `confidence` ve `top3` döndür

## 📝 Git Kuralları

### Commit Message
- Format: `type(scope): description`
- Tipler: `feat`, `fix`, `data`, `model`, `docs`, `refactor`, `test`, `chore`
- Örnekler:
  - `feat(detector): add YOLO training script`
  - `data(classifier): expand label mapping to 40 classes`
  - `model(classifier): add EfficientNet-B0 baseline`

### Gitignore (repo'ya dahil edilmeyecekler)
- `data/` — tüm dataset'ler
- `runs/` — training çıktıları
- `exports/` — model exportları  
- `*.pt`, `*.pth`, `*.onnx`, `*.tflite` — model dosyaları
- `__pycache__/`, `.venv/`, `.env`

## 🧪 Test & Doğrulama

- Data pipeline sonrası: sınıf dağılımı, örnek görüntü kontrolü
- Model eğitimi sonrası: mAP (detection), accuracy/F1 (classification)
- Export sonrası: inference doğrulama (aynı girdi, aynı çıktı)
- API: smoke test ile end-to-end kontrol

## 📦 Kaggle/Colab Uyumluluğu

- Script'ler hem lokal hem Kaggle'da çalışabilmeli
- Environment variable ile path override: `HEALFOAI_DATA_ROOT`, `HEALFOAI_PROJECT_ROOT`
- GPU check: `torch.cuda.is_available()` ile otomatik device seçimi
- Kaggle notebook'ları `notebooks/kaggle/` altında tutulabilir
