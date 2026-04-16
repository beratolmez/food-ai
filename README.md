# Food AI

Bu proje, mobil uygulamaya entegre edilebilecek iki ayrı model üretmek için hazırlanmıştır:
- YOLO tabanlı food detection modeli
- Hafif food classification modeli

## Amaç
İlk fazın hedefi yüksek accuracy değil, mobilde çalıştırılabilir ve export edilebilir bir model hattı kurmaktır.

## Faz 1 çıktıları
- Detection için eğitilebilir YOLO veri yapısı
- Classification için hazırlanmış Food-101 tabanlı veri yapısı
- TFLite export edilebilir classifier
- Daha sonra export edilebilir detector

## Klasör yapısı
- `configs/`: sınıf listeleri ve mapping dosyaları
- `data/`: ham, işlenmiş ve split edilmiş veri
- `scripts/`: veri hazırlama, eğitim ve export scriptleri
- `training/`: model eğitim dosyaları
- `exports/`: mobil için export edilmiş modeller

## Başlangıç adımları
1. `configs/classes_v1.json` dosyasını gözden geçir
2. `configs/label_mapping_v1.json` dosyasını düzenle
3. Food-101 veri setini indir
4. `prepare_food101_classifier.py` scriptini çalıştır
5. İlk classification baseline modelini eğit

## Faz 1 öncelikleri
- Sınıf sözlüğünü sabitle
- Veri hazırlama hattını kur
- Küçük ama çalışan bir baseline classifier üret
- TFLite export al

## Not
Detection ve classification veri hatları ayrı tutulmalıdır.