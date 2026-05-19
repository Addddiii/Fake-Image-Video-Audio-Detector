# ============================================================
# Video Training Script
# EfficientNet-B0 + BiLSTM + 6 channels + 4 landmark features
# ============================================================

from pathlib import Path
import random
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from architectures.video_model import VideoClassifierLSTM

# =========================
# PATHS
# =========================

DATA_DIR = Path(r"D:\Videos\processed")
MODEL_DIR = Path(r"D:\Videos\model")

PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"

FEATURE_CACHE_PATH = DATA_DIR / "video_feature_cache.pt"

BEST_PATH = MODEL_DIR / "video_best.pth"
FINAL_PATH = MODEL_DIR / "video_final.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# SETTINGS
# =========================

BATCH_SIZE = 4
EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

FRAMES_PER_VIDEO = 32
MIN_FRAMES_REQUIRED = 16
BALANCE_SPLITS = True
IMAGE_SIZE = 256
PATIENCE = 7
SEED = 42

LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.5

EXTRA_FEATURE_DIM = 4


# =========================
# SEED / DEVICE
# =========================

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# =========================
# DLIB SETUP
# =========================

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing dlib predictor file: {PREDICTOR_PATH}\n"
        "Put shape_predictor_81_face_landmarks.dat inside dlib_tools."
    )

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))


# =========================
# LANDMARK FEATURE HELPERS
# =========================

def get_landmarks_from_pil(image):
    """
    Detect landmarks from a PIL RGB image.

    Returns:
        Tensor [num_points, 2] or None
    """

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_detector(gray, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = face_predictor(gray, face)

    points = []

    for i in range(shape.num_parts):
        points.append([shape.part(i).x, shape.part(i).y])

    return torch.tensor(points, dtype=torch.float32)


def eye_aspect_ratio(eye_points):
    """
    eye_points should be [6, 2].
    Uses dlib eye points.
    """

    vertical_1 = torch.dist(eye_points[1], eye_points[5])
    vertical_2 = torch.dist(eye_points[2], eye_points[4])
    horizontal = torch.dist(eye_points[0], eye_points[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def mouth_opening_ratio(mouth_points):
    """
    mouth_points should be dlib mouth points 48:68.
    """

    left = mouth_points[0]
    right = mouth_points[6]
    top = mouth_points[3]
    bottom = mouth_points[9]

    vertical = torch.dist(top, bottom)
    horizontal = torch.dist(left, right)

    return vertical / (horizontal + 1e-6)


def compute_landmark_features(landmarks_list):
    """
    Calculates 3 landmark-based features:

    1. mouth/lip movement score
    2. face motion consistency score
    3. eye/blink movement score
    """

    valid = [lm for lm in landmarks_list if lm is not None]

    if len(valid) < 2:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    landmarks = torch.stack(valid)  # [T, N, 2]

    # =========================
    # 1. Mouth/lip movement
    # =========================

    mouth_ratios = []

    for lm in landmarks:
        if lm.shape[0] > 67:
            mouth = lm[48:68]
            mouth_ratios.append(mouth_opening_ratio(mouth))

    if len(mouth_ratios) >= 2:
        mouth_ratios = torch.stack(mouth_ratios)
        mouth_score = mouth_ratios.std()
    else:
        mouth_score = torch.tensor(0.0)

    # =========================
    # 2. Face motion consistency
    # =========================

    # Nose, eyes, mouth corners
    stable_indices = [30, 36, 45, 48, 54]

    if landmarks.shape[1] > max(stable_indices):
        stable_points = landmarks[:, stable_indices, :]  # [T, 5, 2]

        motion = torch.norm(
            stable_points[1:] - stable_points[:-1],
            dim=2
        ).mean(dim=1)

        # Normalize by image size so it is stable
        motion = motion / IMAGE_SIZE

        if motion.numel() >= 2:
            motion_std = motion.std()
            face_motion_consistency = 1.0 / (1.0 + 10.0 * motion_std)
        else:
            face_motion_consistency = torch.tensor(0.0)
    else:
        face_motion_consistency = torch.tensor(0.0)

    # =========================
    # 3. Eye/blink movement
    # =========================

    eye_ratios = []

    for lm in landmarks:
        if lm.shape[0] > 47:
            left_eye = lm[36:42]
            right_eye = lm[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            eye_ratios.append((left_ear + right_ear) / 2.0)

    if len(eye_ratios) >= 2:
        eye_ratios = torch.stack(eye_ratios)
        eye_score = eye_ratios.std()
    else:
        eye_score = torch.tensor(0.0)

    features = torch.stack([
        mouth_score * 10.0,
        face_motion_consistency,
        eye_score * 10.0,
    ])

    return torch.clamp(features, 0.0, 1.0).float()


# =========================
# ARTIFACT FEATURE
# =========================

def compute_laplacian_variance(gray):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=gray.dtype
    ).view(1, 1, 3, 3)

    gray = gray.unsqueeze(0).unsqueeze(0)

    lap = torch.nn.functional.conv2d(
        gray,
        kernel,
        padding=1
    )

    return lap.var()


def compute_artifact_feature(raw_frames):
    """
    Calculates artifact/compression inconsistency from sharpness changes.
    raw_frames:
        Tensor [T, 3, H, W], values 0..1
    """

    sharpness_values = []

    for frame in raw_frames:
        gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        sharpness_values.append(compute_laplacian_variance(gray))

    sharpness_values = torch.stack(sharpness_values)

    if sharpness_values.numel() >= 2:
        artifact_score = sharpness_values.std()
    else:
        artifact_score = torch.tensor(0.0)

    artifact_score = artifact_score * 100.0
    artifact_score = torch.clamp(artifact_score, 0.0, 1.0)

    return artifact_score.float()


def compute_extra_features(raw_frames, landmarks_list):
    """
    Final 4 features:

    0 = mouth/lip movement score
    1 = face motion consistency score
    2 = eye/blink movement score
    3 = artifact/compression inconsistency score
    """

    landmark_features = compute_landmark_features(landmarks_list)
    artifact_feature = compute_artifact_feature(raw_frames).view(1)

    features = torch.cat(
        [landmark_features, artifact_feature],
        dim=0
    )

    return features.float()


# =========================
# DATASET
# =========================

class VideoDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.samples = []

        if not FEATURE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Missing feature cache: {FEATURE_CACHE_PATH}\n"
                "Run precompute_video_features.py first."
            )

        self.feature_cache = torch.load(FEATURE_CACHE_PATH, map_location="cpu")

        self.train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.eval_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.raw_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        
        real_samples = []
        fake_samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = DATA_DIR / split / label_name

            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")

            for video_folder in folder.iterdir():
                if not video_folder.is_dir():
                    continue

                frames = (
                    sorted(video_folder.glob("*.png")) +
                    sorted(video_folder.glob("*.jpg")) +
                    sorted(video_folder.glob("*.jpeg"))
                )

                if len(frames) >= MIN_FRAMES_REQUIRED:
                    key = f"{split}/{label_name}/{video_folder.name}"

                    if key not in self.feature_cache:
                        continue

                    if label == 0:
                        real_samples.append((frames, label, key))
                    else:
                        fake_samples.append((frames, label, key))

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

        self.real_count = sum(1 for _, label, _ in self.samples if label == 0)
        self.fake_count = sum(1 for _, label, _ in self.samples if label == 1)

        print(f"{split}: {len(self.samples)} videos")
        print(f"{split}/real after balance: {self.real_count}")
        print(f"{split}/fake after balance: {self.fake_count}")

    def __len__(self):
        return len(self.samples)

    def pick_frames(self, frames):
        frames = list(frames)

        if len(frames) >= FRAMES_PER_VIDEO:
            step = len(frames) / FRAMES_PER_VIDEO
            indexes = [
                min(int(i * step), len(frames) - 1)
                for i in range(FRAMES_PER_VIDEO)
            ]

            return [frames[i] for i in indexes]

        padded = frames[:]

        while len(padded) < FRAMES_PER_VIDEO:
            padded.append(frames[-1])

        return padded

    def __getitem__(self, index):
        frames, label, key = self.samples[index]
        selected_frames = self.pick_frames(frames)

        model_transform = self.train_tf if self.split == "train" else self.eval_tf

        model_images = []

        for frame_path in selected_frames:
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception:
                image = Image.open(random.choice(frames)).convert("RGB")

            model_images.append(model_transform(image))

        images = torch.stack(model_images)

        # Load precomputed 4 landmark/artifact features
        extra_features = self.feature_cache[key].float()

        # Add motion channels: RGB + frame difference = 6 channels
        motion = torch.zeros_like(images)
        motion[1:] = torch.abs(images[1:] - images[:-1])

        images = torch.cat([images, motion], dim=1)

        label = torch.tensor(label, dtype=torch.long)

        return images, extra_features, label


# =========================
# MODEL
# =========================

def build_model():
    model = VideoClassifierLSTM(
        num_classes=2,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        in_channels=6,
        extra_feature_dim=EXTRA_FEATURE_DIM,
        use_extra_features=True,
    )

    return model


# =========================
# TRAIN / EVAL LOOP
# =========================

def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    scaler=None,
    name="Train"
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

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
        mininterval=1.0
    )

    for videos, extra_features, labels in loop:
        videos = videos.to(device, non_blocking=True)
        extra_features = extra_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda")
            ):
                spatial_logits, temporal_logits, _ = model(
                    videos,
                    extra_features
                )

                loss_temporal = loss_fn(temporal_logits, labels)
                loss_spatial = loss_fn(spatial_logits, labels)

                loss = loss_temporal + 0.5 * loss_spatial

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = temporal_logits.argmax(dim=1)

        loss_sum += loss.item() * videos.size(0)
        correct += (preds == labels).sum().item()
        total += videos.size(0)

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
            bal=f"{bal_acc:.4f}",
            real=f"{real_acc:.4f}",
            fake=f"{fake_acc:.4f}",
        )

    epoch_loss = loss_sum / total
    epoch_acc = correct / total
    real_acc = real_correct / max(1, real_total)
    fake_acc = fake_correct / max(1, fake_total)
    bal_acc = (real_acc + fake_acc) / 2

    return epoch_loss, epoch_acc, bal_acc, real_acc, fake_acc


# =========================
# MAIN
# =========================

def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = VideoDataset("train")
    eval_ds = VideoDataset("eval")
    test_ds = VideoDataset("test")

    loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_settings
    )

    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **loader_settings
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_settings
    )

    model = build_model().to(DEVICE)

    # Class weights are kept, but after balancing they should be close to [1.0, 1.0]
    total_train = train_ds.real_count + train_ds.fake_count

    real_weight = total_train / max(1, 2 * train_ds.real_count)
    fake_weight = total_train / max(1, 2 * train_ds.fake_count)

    class_weights = torch.tensor(
        [real_weight, fake_weight],
        dtype=torch.float32,
        device=DEVICE
    )

    print("Class weights:", class_weights.detach().cpu().tolist())

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    best_bal_acc = 0.0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss, train_acc, train_bal, train_real, train_fake = run_epoch(
            model,
            train_loader,
            loss_fn,
            DEVICE,
            optimizer,
            scaler,
            "Train"
        )

        eval_loss, eval_acc, eval_bal, eval_real, eval_fake = run_epoch(
            model,
            eval_loader,
            loss_fn,
            DEVICE,
            name="Eval"
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

        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        if eval_bal > best_bal_acc:
            best_bal_acc = eval_bal
            wait = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_balanced_acc": best_bal_acc,
                "model": "efficientnet_b0_bilstm_6ch_4_landmark_features",
                "frames_per_video": FRAMES_PER_VIDEO,
                "image_size": IMAGE_SIZE,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "dropout": DROPOUT,
                "in_channels": 6,
                "extra_feature_dim": EXTRA_FEATURE_DIM,
                "use_extra_features": True,
                "extra_features": [
                    "mouth_lip_landmark_movement",
                    "face_landmark_motion_consistency",
                    "eye_blink_landmark_movement",
                    "artifact_compression_inconsistency"
                ],
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
        model,
        test_loader,
        loss_fn,
        DEVICE,
        name="Test"
    )

    print("\nFINAL RESULTS")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Test Acc:          {test_acc:.4f}")
    print(f"Test Balanced Acc: {test_bal:.4f}")
    print(f"Test Real Acc:     {test_real:.4f}")
    print(f"Test Fake Acc:     {test_fake:.4f}")

    torch.save(model.state_dict(), FINAL_PATH)

    print(f"Final model saved to: {FINAL_PATH}")


if __name__ == "__main__":
    main()