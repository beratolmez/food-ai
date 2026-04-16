# Faz 1 hedefi

## Detection
- Model: YOLOv8n
- Amaç: coarse food bbox detection
- Çıktı: best.pt -> TFLite export

## Classification
- Model: MobileNetV3 veya EfficientNet-Lite
- Amaç: crop üstünde top-3 class prediction
- Çıktı: .keras / SavedModel -> .tflite

## Constraint
- İlk sprintte accuracy değil, çalışan export hattı
- Android öncelikli test