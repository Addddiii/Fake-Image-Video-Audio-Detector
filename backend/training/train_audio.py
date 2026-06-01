"""
Train the audio deepfake detection model.
Uses EfficientNet-B0 with log-mel spectrograms and 13 handcrafted audio features.
"""

from pathlib import Path
import random
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from architectures.audio_model import create_audio_detection_model


DATA_DIR = Path(r"D:\audio_processed")
MODEL_DIR = Path(r"D:\audio_model")

BEST_PATH = MODEL_DIR / "audio_best.pth"
FINAL_PATH = MODEL_DIR / "audio_final.pth"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
PATIENCE = 5
IMAGE_SIZE = 224
SEED = 42

EXTRA_FEATURE_DIM = 13
USE_EXTRA_FEATURES = True
BALANCE_SPLITS = False

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


class AudioDataset(Dataset):
    def __init__(self, split):
        self.split = split

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        real_samples = []
        fake_samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = DATA_DIR / split / label_name

            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")

            files = sorted(folder.glob("*.npz"))
            labelled_samples = [(file_path, label) for file_path in files]

            if label == 0:
                real_samples.extend(labelled_samples)
            else:
                fake_samples.extend(labelled_samples)

        print(f"{split}/real before balance: {len(real_samples)}")
        print(f"{split}/fake before balance: {len(fake_samples)}")

        if BALANCE_SPLITS:
            min_count = min(len(real_samples), len(fake_samples))

            random.shuffle(real_samples)
            random.shuffle(fake_samples)

            real_samples = real_samples[:min_count]
            fake_samples = fake_samples[:min_count]

        self.samples = real_samples + fake_samples
        random.shuffle(self.samples)

        self.real_count = sum(1 for _, label in self.samples if label == 0)
        self.fake_count = sum(1 for _, label in self.samples if label == 1)

        print(f"{split}: {len(self.samples)} audio files")
        print(f"{split}/real after balance: {self.real_count}")
        print(f"{split}/fake after balance: {self.fake_count}")

    def __len__(self):
        return len(self.samples)

    def mel_to_image(self, mel):
        mel = mel.astype(np.float32)

        mel_min = mel.min()
        mel_max = mel.max()

        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
        mel = (mel * 255.0).clip(0, 255).astype(np.uint8)

        return Image.fromarray(mel).convert("RGB")

    def __getitem__(self, index):
        npz_path, label = self.samples[index]

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            npz_path, label = random.choice(self.samples)
            data = np.load(npz_path, allow_pickle=True)

        mel = data["mel"]
        extra_features = data["features"].astype(np.float32)

        image = self.mel_to_image(mel)
        transform = self.train_transform if self.split == "train" else self.eval_transform

        image_tensor = transform(image)
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, extra_features, label_tensor


def build_model():
    return create_audio_detection_model(
        num_classes=2,
        pretrained=True,
        dropout=0.4,
        extra_feature_dim=EXTRA_FEATURE_DIM,
        use_extra_features=USE_EXTRA_FEATURES,
    )


def run_epoch(
    model,
    loader,
    loss_fn,
    optimiser=None,
    scaler=None,
    name="Train",
):
    training = optimiser is not None

    model.train() if training else model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    loop = tqdm(
        loader,
        desc=name,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for audio_images, extra_features, labels in loop:
        audio_images = audio_images.to(DEVICE, non_blocking=True)
        extra_features = extra_features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(audio_images, extra_features)
                loss = loss_fn(logits, labels)

            if training:
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

        predictions = logits.argmax(dim=1)

        batch_size = audio_images.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        for prediction, label in zip(predictions, labels):
            if label.item() == 0:
                real_total += 1
                real_correct += int(prediction.item() == label.item())
            else:
                fake_total += 1
                fake_correct += int(prediction.item() == label.item())

        accuracy = correct / total
        real_accuracy = real_correct / max(1, real_total)
        fake_accuracy = fake_correct / max(1, fake_total)
        balanced_accuracy = (real_accuracy + fake_accuracy) / 2

        loop.set_postfix(
            loss=f"{loss_sum / total:.4f}",
            acc=f"{accuracy:.4f}",
            bal=f"{balanced_accuracy:.4f}",
            real=f"{real_accuracy:.4f}",
            fake=f"{fake_accuracy:.4f}",
        )

    epoch_loss = loss_sum / total
    epoch_accuracy = correct / total
    real_accuracy = real_correct / max(1, real_total)
    fake_accuracy = fake_correct / max(1, fake_total)
    balanced_accuracy = (real_accuracy + fake_accuracy) / 2

    return epoch_loss, epoch_accuracy, balanced_accuracy, real_accuracy, fake_accuracy


def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = AudioDataset("train")
    eval_dataset = AudioDataset("eval")
    test_dataset = AudioDataset("test")

    loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_settings)
    eval_loader = DataLoader(eval_dataset, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_settings)

    model = build_model().to(DEVICE)

    total_train = train_dataset.real_count + train_dataset.fake_count

    real_weight = total_train / max(1, 2 * train_dataset.real_count)
    fake_weight = total_train / max(1, 2 * train_dataset.fake_count)

    class_weights = torch.tensor(
        [real_weight, fake_weight],
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Class weights:", class_weights.detach().cpu().tolist())

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        patience=2,
        factor=0.5,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_balanced_accuracy = 0.0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss, train_acc, train_bal, train_real, train_fake = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimiser,
            scaler,
            "Train",
        )

        eval_loss, eval_acc, eval_bal, eval_real, eval_fake = run_epoch(
            model,
            eval_loader,
            loss_fn,
            name="Eval",
        )

        scheduler.step(eval_bal)

        print(
            f"\nTrain Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"Bal: {train_bal:.4f} | "
            f"Real: {train_real:.4f} | "
            f"Fake: {train_fake:.4f}"
        )

        print(
            f"Eval  Loss: {eval_loss:.4f} | "
            f"Acc: {eval_acc:.4f} | "
            f"Bal: {eval_bal:.4f} | "
            f"Real: {eval_real:.4f} | "
            f"Fake: {eval_fake:.4f}"
        )

        print(f"LR: {optimiser.param_groups[0]['lr']:.8f}")

        if eval_bal > best_balanced_accuracy:
            best_balanced_accuracy = eval_bal
            wait = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_balanced_acc": best_balanced_accuracy,
                "model": "efficientnet_b0_audio_13_features",
                "image_size": IMAGE_SIZE,
                "dropout": 0.4,
                "extra_feature_dim": EXTRA_FEATURE_DIM,
                "use_extra_features": USE_EXTRA_FEATURES,
                "extra_features": [
                    "rms_mean",
                    "rms_std",
                    "silence_ratio",
                    "speech_rate_proxy",
                    "pitch_mean",
                    "pitch_std",
                    "pitch_jitter",
                    "spectral_flatness_mean",
                    "spectral_flatness_std",
                    "centroid_mean",
                    "centroid_std",
                    "zcr_mean",
                    "zcr_std",
                ],
                "classes": ["real", "fake"],
            }

            torch.save(checkpoint, BEST_PATH)
            print(f"Saved best model: {BEST_PATH}")
        else:
            wait += 1
            print(f"No improvement ({wait}/{PATIENCE})")

        if wait >= PATIENCE:
            print("Early stopping")
            break

    print("\n===== TEST =====")

    checkpoint = torch.load(
        BEST_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_bal, test_real, test_fake = run_epoch(
        model,
        test_loader,
        loss_fn,
        name="Test",
    )

    print("\nFINAL RESULTS")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Test Acc:          {test_acc:.4f}")
    print(f"Test Balanced Acc: {test_bal:.4f}")
    print(f"Test Real Acc:     {test_real:.4f}")
    print(f"Test Fake Acc:     {test_fake:.4f}")

    torch.save(model.state_dict(), FINAL_PATH)

    print(f"Best model saved to:  {BEST_PATH}")
    print(f"Final model saved to: {FINAL_PATH}")


if __name__ == "__main__":
    main()