food-ai/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ classes_v1.json
│  ├─ label_mapping_v1.json
│  ├─ detector_data.yaml
│  └─ training_plan.md
├─ data/
│  ├─ classification/
│  │  ├─ raw/
│  │  │  └─ food101/
│  │  ├─ interim/
│  │  ├─ processed/
│  │  └─ splits/
│  └─ detection/
│     ├─ raw/
│     │  └─ uec_food256/
│     ├─ interim/
│     ├─ yolo/
│     │  ├─ images/
│     │  │  ├─ train/
│     │  │  ├─ val/
│     │  │  └─ test/
│     │  └─ labels/
│     │     ├─ train/
│     │     ├─ val/
│     │     └─ test/
│     └─ crops/
├─ notebooks/
│  ├─ 01_dataset_inspection.ipynb
│  ├─ 02_label_mapping_check.ipynb
│  └─ 03_baseline_training.ipynb
├─ scripts/
│  ├─ download_food101.py
│  ├─ prepare_food101_classifier.py
│  ├─ prepare_uec_for_yolo.py
│  ├─ build_label_mapping.py
│  ├─ generate_crops_from_yolo.py
│  ├─ train_classifier.py
│  ├─ export_classifier_tflite.py
│  └─ validate_exports.py
├─ training/
│  ├─ detector/
│  └─ classifier/
├─ runs/
│  ├─ detector/
│  └─ classifier/
└─ exports/
   ├─ detector/
   └─ classifier/