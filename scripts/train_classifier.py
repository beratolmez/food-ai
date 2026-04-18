"""Train PyTorch classification model.

Features 2-stage transfer learning:
1. Feature Extraction: Freezes backbone, trains only the custom classifier head.
2. Fine-tuning: Unfreezes the whole network, trains with a smaller learning rate.

This script should be run on a GPU environment (Colab/Kaggle).

Usage:
    python scripts/train_classifier.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.classifier_dataset import FoodClassifierDataset, get_transforms
from src.models.classifier import create_classifier, freeze_backbone, unfreeze_all

# Default to Kaggle working dir if inside Kaggle, else local runs
KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_RUNS_DIR = KAGGLE_WORKING / "runs" if KAGGLE_WORKING.exists() else PROJECT_ROOT / "runs"

# Path to the data inside Kaggle (you might need to adjust this depending on how you mount it)
DEFAULT_DATA_ROOT = Path("/kaggle/input/healfoai-data/splits/classification") if KAGGLE_WORKING.exists() else PROJECT_ROOT / "data" / "splits" / "classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Food Classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "classifier_train_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to classification splits directory",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_RUNS_DIR / "classifier"),
        help="Project directory to save runs",
    )
    parser.add_argument("--name", type=str, default="effnet_b0_m1", help="Experiment name")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Initiating PyTorch Classifier Training")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Model      : {config['model_name']}")
    print(f"Data Dir   : {args.data_dir}")
    print(f"Output Dir : {run_dir}")
    print("=" * 60)

    # 1. Prepare classes mapping
    train_dir = args.data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # Save class mapping for reference later
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
        
    print(f"Found {num_classes} classes.")

    # 2. Datasets & Dataloaders
    img_size = config["image_size"]
    train_transform = get_transforms(is_train=True, image_size=img_size)
    val_transform = get_transforms(is_train=False, image_size=img_size)

    train_ds = FoodClassifierDataset(args.data_dir / "train", class_to_idx, train_transform)
    val_ds = FoodClassifierDataset(args.data_dir / "val", class_to_idx, val_transform)

    workers = config.get("workers", 4)
    pin_mem = config.get("pin_memory", True)
    
    train_loader = DataLoader(
        train_ds, batch_size=config["stage1"]["batch_size"], shuffle=True, 
        num_workers=workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["stage1"]["batch_size"], shuffle=False, 
        num_workers=workers, pin_memory=pin_mem
    )

    # 3. Create Model
    model = create_classifier(config["model_name"], num_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # ---------------------------------------------------------
    # STAGE 1: Feature Extraction
    # ---------------------------------------------------------
    print("\n[STAGE 1] Feature Extraction (Training head only)")
    freeze_backbone(model, config["model_name"])
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["stage1"]["learning_rate"],
        weight_decay=config["stage1"]["weight_decay"]
    )

    for epoch in range(config["stage1"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['stage1']['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print("  --> Saved new best model!")

    # ---------------------------------------------------------
    # STAGE 2: Fine-Tuning
    # ---------------------------------------------------------
    print("\n[STAGE 2] Fine-Tuning (Training all layers)")
    # Load best weights from Stage 1 before starting Stage 2
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
    unfreeze_all(model)
    
    # Update dataloader batch size for fine-tuning
    train_loader = DataLoader(
        train_ds, batch_size=config["stage2"]["batch_size"], shuffle=True, 
        num_workers=workers, pin_memory=pin_mem
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["stage2"]["learning_rate"],
        weight_decay=config["stage2"]["weight_decay"]
    )
    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    for epoch in range(config["stage2"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['stage2']['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print("  --> Saved new best model!")

    # Save final model state
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    
    print("\nTraining completed successfully!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved into: {run_dir}")


if __name__ == "__main__":
    main()