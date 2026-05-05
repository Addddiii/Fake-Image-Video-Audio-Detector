# ============================================================
# Video Training Script
# ============================================================

from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(r"D:\Videos_Processed")
MODEL_DIR = Path(r"D:\video\model")

BEST_PATH = MODEL_DIR / "video_best.pth"
FINAL_PATH = MODEL_DIR / "video_final.pth"

BATCH_SIZE = 12
EPOCHS = 25
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
FRAMES_PER_VIDEO = 12
IMAGE_SIZE = 224
PATIENCE = 7
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


class VideoDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.samples = []

        self.train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.eval_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = DATA_DIR / split / label_name

            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")

            for video_folder in folder.iterdir():
                if video_folder.is_dir():
                    frames = sorted(video_folder.glob("*.jpg"))

                    if len(frames) >= FRAMES_PER_VIDEO:
                        self.samples.append((frames, label))

        print(f"{split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def pick_frames(self, frames):
        if self.split == "train":
            return sorted(random.sample(frames, FRAMES_PER_VIDEO))

        step = len(frames) / FRAMES_PER_VIDEO
        indexes = [min(int(i * step), len(frames) - 1) for i in range(FRAMES_PER_VIDEO)]
        return [frames[i] for i in indexes]

    def __getitem__(self, index):
        frames, label = self.samples[index]
        selected_frames = self.pick_frames(frames)

        transform = self.train_tf if self.split == "train" else self.eval_tf

        images = []
        for frame_path in selected_frames:
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception:
                image = Image.open(random.choice(frames)).convert("RGB")

            images.append(transform(image))

        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.long)

        return images, label


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2)
    )

    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, name="Train"):
    training = optimizer is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    real_correct, real_total = 0, 0
    fake_correct, fake_total = 0, 0

    loop = tqdm(loader, desc=name, dynamic_ncols=True, mininterval=1.0)

    for videos, labels in loop:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size, frames, c, h, w = videos.shape
        videos = videos.view(batch_size * frames, c, h, w)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(videos)
                outputs = outputs.view(batch_size, frames, 2).mean(dim=1)
                loss = loss_fn(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = outputs.argmax(dim=1)

        loss_sum += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size

        for pred, label in zip(preds, labels):
            if label.item() == 0:
                real_total += 1
                real_correct += int(pred.item() == label.item())
            else:
                fake_total += 1
                fake_correct += int(pred.item() == label.item())

        acc = correct / total
        real_acc = real_correct / max(1, real_total)
        fake_acc = fake_correct / max(1, fake_total)
        bal_acc = (real_acc + fake_acc) / 2

        loop.set_postfix(
            loss=f"{loss_sum / total:.4f}",
            acc=f"{acc:.4f}",
            bal=f"{bal_acc:.4f}"
        )

    loss = loss_sum / total
    acc = correct / total
    real_acc = real_correct / max(1, real_total)
    fake_acc = fake_correct / max(1, fake_total)
    bal_acc = (real_acc + fake_acc) / 2

    return loss, acc, bal_acc, real_acc, fake_acc


def main():
    train_ds = VideoDataset("train")
    eval_ds = VideoDataset("eval")
    test_ds = VideoDataset("test")

    loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_settings)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_settings)

    model = build_model().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_bal_acc = 0.0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss, train_acc, train_bal, train_real, train_fake = run_epoch(
            model, train_loader, loss_fn, DEVICE, optimizer, scaler, "Train"
        )

        eval_loss, eval_acc, eval_bal, eval_real, eval_fake = run_epoch(
            model, eval_loader, loss_fn, DEVICE, name="Eval"
        )

        scheduler.step(eval_bal)

        print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal: {train_bal:.4f}")
        print(f"Eval  Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | Bal: {eval_bal:.4f}")
        print(f"Eval Real Acc: {eval_real:.4f} | Eval Fake Acc: {eval_fake:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        if eval_bal > best_bal_acc:
            best_bal_acc = eval_bal
            wait = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_balanced_acc": best_bal_acc,
                "model": "efficientnet_b0",
                "frames_per_video": FRAMES_PER_VIDEO,
                "image_size": IMAGE_SIZE,
                "classes": ["real", "fake"],
            }, BEST_PATH)

            print(f"Saved best model: {BEST_PATH}")
        else:
            wait += 1
            print(f"No improvement ({wait}/{PATIENCE})")

        if wait >= PATIENCE:
            print("Early stopping")
            break

    print("\n===== TEST =====")

    checkpoint = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_bal, test_real, test_fake = run_epoch(
        model, test_loader, loss_fn, DEVICE, name="Test"
    )

    print("\nFINAL RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Balanced Acc: {test_bal:.4f}")
    print(f"Test Real Acc: {test_real:.4f}")
    print(f"Test Fake Acc: {test_fake:.4f}")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Final model saved to: {FINAL_PATH}")


if __name__ == "__main__":
    main()