# HealfoAI ML — Görev Listesi

> Bu dosya projenin adım adım ilerlemesini takip eder.
> Görevler tamamlandıkça `[x]` ile işaretlenir.

---

## Milestone 1: Proje Altyapısı & Veri Hazırlama (Local)

### 1.1 Proje Altyapısı
- [x] `AGENTS.md` oluştur
- [x] `requirements.txt` güncelle (PyTorch-only, pinned versions)
- [x] `.gitignore` güncelle (runs/, exports/, model dosyaları)
- [x] Klasör yapısı oluştur (`src/`, `api/`, `evaluation/`)
- [x] `src/utils/logger.py` — logging utility
- [x] `configs/classes_v2.json` — 42 sınıf + "other" (toplam 43)
- [x] `configs/label_mapping_v2.json` — Food-101 + UECFOOD256 mapping
- [x] `configs/detector_data.yaml` — tek sınıf "food" YOLO config

### 1.2 UECFOOD256 → YOLO Format Dönüşümü (Local)
- [x] `scripts/prepare_uec_for_yolo.py` yaz
- [x] `scripts/validate_yolo_data.py` yaz
- [x] Dönüşüm scriptini çalıştır ve doğrula
- [x] Dönüşüm sonuçlarını incele (bbox örnekleri, split dağılımı)

### 1.3 Food-101 Classification Veri Hazırlama (Local)
- [x] `scripts/prepare_food101_classifier.py` güncelle (v2 mapping, val split)
- [x] Classifier veri hazırlama scriptini çalıştır
- [x] Sınıf dağılımını incele, dengesizlik analizi yap

---

## Milestone 2: YOLO Detection Modeli (Kaggle/Colab)

### 2.1 Training
- [ ] `scripts/train_detector.py` — YOLO training scripti
- [ ] Kaggle notebook oluştur: YOLO training
- [ ] Dataset'i Kaggle'a yükle
- [ ] Training çalıştır (YOLOv8n, tek sınıf "food")

### 2.2 Değerlendirme
- [ ] `scripts/evaluate_detector.py` — mAP, confusion matrix
- [ ] Sonuçları değerlendir (hedef: mAP@0.5 > 0.60)

### 2.3 Export
- [ ] `scripts/export_detector.py` — ONNX export
- [ ] Export doğrulama (aynı girdi → aynı çıktı)

---

## Milestone 3: Classification Modeli (Kaggle/Colab)

### 3.1 Model & Dataset Kodu
- [ ] `src/models/classifier.py` — EfficientNet-B0 / MobileNetV3 model tanımı
- [ ] `src/data/classifier_dataset.py` — PyTorch Dataset + augmentation
- [ ] `configs/classifier_train_config.yaml` — training hyperparameters

### 3.2 Training
- [ ] `scripts/train_classifier.py` — tamamen yeniden yaz (PyTorch)
- [ ] Kaggle notebook oluştur: Classifier training
- [ ] Dataset'i Kaggle'a yükle
- [ ] Aşama 1: Feature extraction (backbone frozen, 5-10 epoch)
- [ ] Aşama 2: Fine-tuning (son N katman açık, 15-20 epoch)

### 3.3 Değerlendirme
- [ ] `scripts/evaluate_classifier.py` — accuracy, F1, confusion matrix
- [ ] Sonuçları değerlendir (hedef: Top-1 > %70, Top-3 > %85)

### 3.4 Export
- [ ] `scripts/export_classifier.py` — ONNX export
- [ ] Export doğrulama

---

## Milestone 4: End-to-End Inference Pipeline

### 4.1 Inference Modülleri
- [ ] `src/inference/detector.py` — YOLO inference wrapper
- [ ] `src/inference/classifier.py` — Classifier inference wrapper
- [ ] `src/inference/pipeline.py` — Detect → Crop → Classify pipeline

### 4.2 Test & Benchmark
- [ ] Test scripti: örnek görüntülerle end-to-end test
- [ ] Latency benchmark
- [ ] Sonuçları raporla

---

## Milestone 5: İyileştirme & İterasyon

### 5.1 Model İyileştirme
- [ ] Yanlış tahmin analizi
- [ ] Ek veri toplama (zayıf sınıflar için)
- [ ] UECFOOD256 crop'larını classification'a ekle
- [ ] Re-training ve karşılaştırma

### 5.2 Kalori Entegrasyonu
- [ ] `configs/nutrition_db.json` — besin-kalori veritabanı
- [ ] Pipeline'a kalori bilgisi ekle

### 5.3 Mevcut Backend'e Entegrasyon
- [ ] Model dosyalarını backend projesine aktar
- [ ] Inference pipeline'ı backend'e entegre et
- [ ] End-to-end test (mobil app → backend → model → sonuç)
