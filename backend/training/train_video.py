"""
Train the video deepfake detection model.

Uses EfficientNet-B0, BiLSTM, RGB + motion channels, and four cached video features.
"""

from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from architectures.video_model import VideoClassifierLSTM

DATA_DIR = Path(r"D:\videos_processed")
MODEL_DIR = Path(r"D:\videos_model")
FEATURE_CACHE_PATH = DATA_DIR / "video_feature_cache.pt"

BEST_PATH = MODEL_DIR / "video_best.pth"
FINAL_PATH = MODEL_DIR / "video_final.pth"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

FRAMES_PER_VIDEO = 32
MIN_FRAMES_REQUIRED = 16
IMAGE_SIZE = 256
PATIENCE = 7
SEED = 42

LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.5

EXTRA_FEATURE_DIM = 4
BALANCE_SPLITS = True

FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg"}

VIDEO_FEATURE_NAMES = [
    "mouth_lip_movement",
    "face_motion_consistency",
    "eye_blink_movement",
    "artifact_compression_inconsistency",
]

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


class VideoDataset(Dataset):
    """
    Loads processed video frame folders and cached handcrafted video features.
    """

    def __init__(self, split):
        self.split = split
        self.feature_cache = self._load_feature_cache()
        self.train_transform, self.eval_transform = self._build_transforms()
        self.samples = self._load_samples()

        self.real_count = sum(1 for _, label, _ in self.samples if label == 0)
        self.fake_count = sum(1 for _, label, _ in self.samples if label == 1)

        print(f"{split}: {len(self.samples)} videos")
        print(f"{split}/real after balance: {self.real_count}")
        print(f"{split}/fake after balance: {self.fake_count}")

    def _load_feature_cache(self):
        """
        Load precomputed video feature vectors.
        """
        if not FEATURE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Missing feature cache: {FEATURE_CACHE_PATH}\n"
                "Run precompute_video_features.py first."
            )

        return torch.load(
            FEATURE_CACHE_PATH,
            map_location="cpu",
            weights_only=True,
        )

    def _build_transforms(self):
        """
        Create training and evaluation transforms for video frames.
        """
        train_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        return train_transform, eval_transform

    def _collect_frame_paths(self, video_folder):
        """
        Collect valid frame paths from one processed video folder.
        """
        return [
            file_path
            for file_path in sorted(video_folder.iterdir())
            if file_path.is_file() and file_path.suffix.lower() in FRAME_EXTENSIONS
        ]

    def _collect_label_samples(self, label_name, label):
        """
        Collect valid video samples for one label.
        """
        folder = DATA_DIR / self.split / label_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        samples = []

        for video_folder in folder.iterdir():
            if not video_folder.is_dir():
                continue

            frames = self._collect_frame_paths(video_folder)

            if len(frames) < MIN_FRAMES_REQUIRED:
                continue

            key = f"{self.split}/{label_name}/{video_folder.name}"

            if key not in self.feature_cache:
                continue

            samples.append((frames, label, key))

        return samples

    def _load_samples(self):
        """
        Load real and fake video samples for the selected split.
        """
        real_samples = self._collect_label_samples("real", 0)
        fake_samples = self._collect_label_samples("fake", 1)

        print(f"{self.split}/real before balance: {len(real_samples)}")
        print(f"{self.split}/fake before balance: {len(fake_samples)}")

        if BALANCE_SPLITS:
            min_count = min(len(real_samples), len(fake_samples))

            random.shuffle(real_samples)
            random.shuffle(fake_samples)

            real_samples = real_samples[:min_count]
            fake_samples = fake_samples[:min_count]

        samples = real_samples + fake_samples
        random.shuffle(samples)

        return samples

    def __len__(self):
        return len(self.samples)

    def pick_frames(self, frames):
        """
        Select exactly FRAMES_PER_VIDEO evenly spaced frames.
        """
        frames = list(frames)

        if len(frames) >= FRAMES_PER_VIDEO:
            step = len(frames) / FRAMES_PER_VIDEO
            indexes = [
                min(int(index * step), len(frames) - 1)
                for index in range(FRAMES_PER_VIDEO)
            ]

            return [frames[index] for index in indexes]

        padded_frames = frames[:]

        while len(padded_frames) < FRAMES_PER_VIDEO:
            padded_frames.append(frames[-1])

        return padded_frames

    def __getitem__(self, index):
        frames, label, key = self.samples[index]
        selected_frames = self.pick_frames(frames)

        transform = self.train_transform if self.split == "train" else self.eval_transform

        images = []

        for frame_path in selected_frames:
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception:
                image = Image.open(random.choice(frames)).convert("RGB")

            images.append(transform(image))

        images = torch.stack(images)
        extra_features = self.feature_cache[key].float()

        motion = torch.zeros_like(images)
        motion[1:] = torch.abs(images[1:] - images[:-1])

        images = torch.cat([images, motion], dim=1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return images, extra_features, label_tensor


def build_model():
    """
    Create the EfficientNet-B0 + BiLSTM video classification model.
    """
    return VideoClassifierLSTM(
        num_classes=2,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        in_channels=6,
        extra_feature_dim=EXTRA_FEATURE_DIM,
        use_extra_features=True,
    )


def run_epoch(model, loader, loss_fn, optimiser=None, scaler=None, name="Train"):
    """
    Run one training, validation, or testing epoch.
    """
    training = optimiser is not None
    model.train() if training else model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    loop = tqdm(loader, desc=name, dynamic_ncols=True, mininterval=1.0)

    for videos, extra_features, labels in loop:
        videos = videos.to(DEVICE, non_blocking=True)
        extra_features = extra_features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                spatial_logits, temporal_logits, _ = model(videos, extra_features)

                temporal_loss = loss_fn(temporal_logits, labels)
                spatial_loss = loss_fn(spatial_logits, labels)
                loss = temporal_loss + 0.5 * spatial_loss

            if training:
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

        predictions = temporal_logits.argmax(dim=1)

        batch_size = videos.size(0)
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


def create_checkpoint(model, epoch, best_balanced_accuracy):
    """
    Create a checkpoint dictionary containing the model state and metadata.
    """
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_balanced_acc": best_balanced_accuracy,
        "model": "efficientnet_b0_bilstm_6ch_4_video_features",
        "frames_per_video": FRAMES_PER_VIDEO,
        "image_size": IMAGE_SIZE,
        "lstm_hidden": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "dropout": DROPOUT,
        "in_channels": 6,
        "extra_feature_dim": EXTRA_FEATURE_DIM,
        "use_extra_features": True,
        "extra_features": VIDEO_FEATURE_NAMES,
        "classes": ["real", "fake"],
    }


def print_epoch_summary(
    train_loss,
    train_acc,
    train_bal,
    train_real,
    train_fake,
    eval_loss,
    eval_acc,
    eval_bal,
    eval_real,
    eval_fake,
    optimiser,
):
    """
    Print training and validation metrics for one epoch.
    """
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


def main():
    """
    Train, validate, test, and save the video detection model.
    """
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = VideoDataset("train")
    eval_dataset = VideoDataset("eval")
    test_dataset = VideoDataset("test")

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
        patience=3,
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

        print_epoch_summary(
            train_loss,
            train_acc,
            train_bal,
            train_real,
            train_fake,
            eval_loss,
            eval_acc,
            eval_bal,
            eval_real,
            eval_fake,
            optimiser,
        )

        if eval_bal > best_balanced_accuracy:
            best_balanced_accuracy = eval_bal
            wait = 0

            checkpoint = create_checkpoint(model, epoch, best_balanced_accuracy)
            torch.save(checkpoint, BEST_PATH)

            print(f"Saved best model: {BEST_PATH}")
        else:
            wait += 1
            print(f"No improvement ({wait}/{PATIENCE})")

        if wait >= PATIENCE:
            print("Early stopping")
            break

    print("\n===== TEST =====")

    checkpoint = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
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