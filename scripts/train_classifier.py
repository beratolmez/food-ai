from __future__ import annotations
def get_callbacks() -> list[keras.callbacks.Callback]:
    ensure_dir(MODEL_DIR)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(MODEL_DIR / "training_log.csv")),
    ]


def fine_tune_model(model: keras.Model) -> keras.Model:
    base_model = model.get_layer(index=2)
    if not isinstance(base_model, keras.Model):
        raise TrainingConfigError("Base model layer could not be resolved for fine-tuning.")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
    return model


def save_label_metadata(class_names: list[str]) -> None:
    ensure_dir(EXPORT_DIR)
    metadata_path = EXPORT_DIR / "classifier_labels_v1.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)


def main() -> None:
    expected_classes = load_class_names()
    train_ds, val_ds, found_classes = build_datasets()
    verify_class_alignment(found_classes, expected_classes)
    train_ds, val_ds = optimize_datasets(train_ds, val_ds)

    model = build_model(num_classes=len(found_classes))
    compile_model(model, learning_rate=LEARNING_RATE)

    print("
=== Initial Training ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=get_callbacks(),
    )

    print("
=== Fine-Tuning ===")
    model = fine_tune_model(model)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
    )

    ensure_dir(MODEL_DIR)
    final_model_path = MODEL_DIR / "final_model.keras"
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    save_label_metadata(found_classes)
    print(f"Saved label metadata to: {EXPORT_DIR / 'classifier_labels_v1.json'}")

    results = model.evaluate(val_ds, verbose=0)
    print("
=== Final Evaluation ===")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()