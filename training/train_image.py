import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # =========================
    # PATHS
    # =========================
    DATA_DIR = r"D:\FakeDetection\processed_datasets\image"
    MODEL_DIR = r"D:\FakeDetection\models\image"
    MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.pth")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # SETTINGS
    # =========================
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 5
    LABEL_SMOOTHING = 0.1

    # =========================
    # DEVICE
    # =========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("Running on CPU (slower)")

    # =========================
    # TRANSFORMS
    # =========================
    # Your images are already 224x224, but RandomResizedCrop still helps
    # by giving slightly varied views during training.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASETS
    # =========================
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    eval_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "eval"),
        transform=eval_test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=eval_test_transform
    )

    print("Classes:", train_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Eval samples:", len(eval_dataset))
    print("Test samples:", len(test_dataset))

    # =========================
    # DATALOADERS
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    # =========================
    # MODEL
    # =========================
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    model = model.to(DEVICE)

    # =========================
    # LOSS / OPTIMIZER / SCHEDULER
    # =========================
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )

    # Mixed precision for GPU speed
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # =========================
    # FUNCTIONS
    # =========================
    def run_epoch(loader, training=False):
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            _, preds = torch.max(outputs, 1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    # =========================
    # TRAINING LOOP
    # =========================
    best_eval_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n========== Epoch {epoch + 1}/{NUM_EPOCHS} ==========")
        print(f"Learning Rate: {current_lr:.8f}")

        train_loss, train_acc = run_epoch(train_loader, training=True)
        eval_loss, eval_acc = run_epoch(eval_loader, training=False)

        scheduler.step(eval_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Eval  Loss: {eval_loss:.4f} | Eval  Acc: {eval_acc:.4f}")

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (NUM_EPOCHS - epoch - 1)

        print(f"Epoch Time: {epoch_time / 60:.2f} min")
        print(f"Estimated Remaining: {remaining_time / 60:.2f} min")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_PATH)
            print(f"Saved best model to: {MODEL_PATH}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # =========================
    # FINAL TEST
    # =========================
    model.load_state_dict(best_model_weights)
    test_loss, test_acc = run_epoch(test_loader, training=False)

    total_time = time.time() - start_time

    print("\n========== FINAL RESULTS ==========")
    print(f"Best Eval Loss: {best_eval_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Total Training Time: {total_time / 60:.2f} min")
    print("Training complete.")


if __name__ == "__main__":
    main()