# HealfoAI ML — Food Detection & Classification Pipeline

Mobil kalori takip uygulaması için besin tanıma ML pipeline'ı.

## 🎯 Pipeline Akışı

```
Fotoğraf → YOLO Detection (food bbox) → Crop → Classification (besin türü) → Sonuç + Kalori
```

## 🏗️ Mimari

| Katman | Model | Detay |
|--------|-------|-------|
| Detection | YOLOv8n | Tek sınıf ("food"), 640×640 |
| Classification | EfficientNet-B0 / MobileNetV3 | 42 sınıf + "other", 224×224 |
| Framework | PyTorch | Tek framework |
| Export | ONNX | Backend inference |

## 📁 Proje Yapısı

```
configs/   → Sınıf listeleri, mapping'ler, training config'leri
data/      → Ham ve işlenmiş veri (gitignore'da)
src/       → Ana kaynak kodu (modüller, inference pipeline)
scripts/   → Çalıştırılabilir scriptler (prepare, train, evaluate, export)
notebooks/ → EDA ve Kaggle training notebook'ları
runs/      → Training çıktıları (gitignore'da)
exports/   → Export edilmiş modeller (gitignore'da)
evaluation/→ Değerlendirme raporları
```

## 🚀 Hızlı Başlangıç

### 1. Veri Hazırlama (Local)
```bash
# UECFOOD256 → YOLO format (detection verisi)
python scripts/prepare_uec_for_yolo.py

# Food-101 → Classification splits
python scripts/prepare_food101_classifier.py

# YOLO veri doğrulama
python scripts/validate_yolo_data.py
```

### 2. Model Eğitimi (Kaggle/Colab)
```bash
# Detection (YOLO)
python scripts/train_detector.py

# Classification (EfficientNet-B0)
python scripts/train_classifier.py
```

### 3. Değerlendirme & Export
```bash
python scripts/evaluate_detector.py
python scripts/evaluate_classifier.py
python scripts/export_detector.py
python scripts/export_classifier.py
```

## 📊 Mevcut Veri Setleri

- **Food-101**: 101 sınıf, ~101K görüntü → 42 hedef sınıfa map'lenir
- **UECFOOD256**: 256 sınıf, bbox annotation'lar → tek sınıf "food" detection

## 📝 Görev Takibi

Proje ilerleme durumu için [TASKS.md](TASKS.md) dosyasına bakın.

## 📦 Gereksinimler

```bash
pip install -r requirements.txt
```