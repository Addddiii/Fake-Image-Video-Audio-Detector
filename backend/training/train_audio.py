# ============================================================
# Audio Training Script
# ============================================================

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

DATA_ROOT = Path(r"D:\audio\processed")
MODEL_DIR = Path(r"D:\audio\model")

BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 4

BEST_MODEL_PATH = MODEL_DIR / "best_audio_model.pth"
FINAL_MODEL_PATH = MODEL_DIR / "final_audio_model.pth"


class AudioDataset(Dataset):
    def __init__(self, folder):
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = Path(folder) / label_name

            if not label_dir.exists():
                continue

            for file_path in label_dir.glob("*.npy"):
                self.samples.append((file_path, label))

        np.random.shuffle(self.samples)
        print(f"{folder}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        x = np.load(file_path).astype(np.float32)

        # normalize spectrogram
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)

        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

        # resize for EfficientNet
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        x = x.squeeze(0)
        x = x.repeat(3, 1, 1)

        y = torch.tensor(label, dtype=torch.long)
        return x, y


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, name="Train"):
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=name, dynamic_ncols=True, mininterval=1.0)

    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(x)
                loss = loss_fn(outputs, y)

            if training:
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = outputs.argmax(dim=1)

        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        correct += (preds == y).sum().item()
        total += batch_size

        loop.set_postfix(
            loss=f"{loss_sum / total:.4f}",
            acc=f"{correct / total:.4f}"
        )

    return loss_sum / total, correct / total


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_ds = AudioDataset(DATA_ROOT / "train")
    eval_ds = AudioDataset(DATA_ROOT / "eval")
    test_ds = AudioDataset(DATA_ROOT / "test")

    loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_settings)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_settings)

    model = build_model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_model = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, device, optimizer, scaler, "Train"
        )

        eval_loss, eval_acc = run_epoch(
            model, eval_loader, loss_fn, device, name="Eval"
        )

        scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Eval  Loss: {eval_loss:.4f} | Eval  Acc: {eval_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_model,
                "epoch": epoch,
                "best_eval_acc": best_acc,
                "model": "efficientnet_b0",
                "classes": ["real", "fake"],
            }, BEST_MODEL_PATH)

            print(f"Saved best model: {BEST_MODEL_PATH}")

    if best_model is not None:
        model.load_state_dict(best_model)

    print("\n===== TESTING BEST MODEL =====")

    test_loss, test_acc = run_epoch(
        model, test_loader, loss_fn, device, name="Test"
    )

    print("\nFINAL TEST RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()