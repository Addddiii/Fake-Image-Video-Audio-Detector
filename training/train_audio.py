import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

DATA_ROOT = r"D:\audio\processed"
MODEL_DIR = r"D:\audio\model"

BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 4


class NpyDataset(Dataset):
    def __init__(self, folder):
        self.files = []
        self.labels = []

        for label, idx in [("real", 0), ("fake", 1)]:
            label_dir = Path(folder) / label
            for f in label_dir.glob("*.npy"):
                self.files.append(f)
                self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = np.load(self.files[i]).astype(np.float32)

        x = (x - np.mean(x)) / (np.std(x) + 1e-6)

        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        x = x.squeeze(0)           # (1,224,224)
        x = x.repeat(3, 1, 1)      # (3,224,224)

        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = loss_fn(out, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = NpyDataset(Path(DATA_ROOT) / "train")
    val_ds   = NpyDataset(Path(DATA_ROOT) / "eval")
    test_ds  = NpyDataset(Path(DATA_ROOT) / "test")

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_model = None

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    print("Starting training...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, Path(MODEL_DIR) / "best_audio_model.pth")
            print("Saved best model")

        print("-" * 40)

    model.load_state_dict(best_model)

    test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)

    print("\nFINAL TEST RESULT")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()