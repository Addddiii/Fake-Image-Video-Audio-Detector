"""
Train the image deepfake detection model.

Uses EfficientNet-B0 with four handcrafted image features.
"""

from pathlib import Path
import random
import sys

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from architectures.image_model import create_fake_detection_model

DATA_DIR = Path(r"D:\images_processed")
MODEL_DIR = Path(r"D:\images_model")

BEST_PATH = MODEL_DIR / "image_best.pth"
FINAL_PATH = MODEL_DIR / "image_final.pth"
PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
PATIENCE = 5
IMAGE_SIZE = 224
SEED = 42

EXTRA_FEATURE_DIM = 4
USE_EXTRA_FEATURES = True
BALANCE_SPLITS = False

IMAGE_FEATURE_NAMES = [
    "face_landmark_quality",
    "face_symmetry_score",
    "eye_mouth_artifact_score",
    "texture_frequency_artifact_score",
]

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(f"Missing dlib predictor file: {PREDICTOR_PATH}")

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))


def get_landmarks(image_pil):
    """
    Detect the largest face and return its facial landmarks.
    """
    image = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = face_predictor(gray, face)

    points = [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]

    return torch.tensor(points, dtype=torch.float32)


def face_landmark_quality(landmarks):
    """
    Estimate face quality based on face size and centre position.
    """
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    x_min = landmarks[:, 0].min()
    x_max = landmarks[:, 0].max()
    y_min = landmarks[:, 1].min()
    y_max = landmarks[:, 1].max()

    face_width = x_max - x_min
    face_height = y_max - y_min

    size_score = torch.clamp(
        (face_width * face_height) / (IMAGE_SIZE * IMAGE_SIZE),
        0.0,
        1.0,
    )

    centre_x = (x_min + x_max) / 2.0
    centre_y = (y_min + y_max) / 2.0

    centre_distance = torch.sqrt(
        ((centre_x - IMAGE_SIZE / 2) / IMAGE_SIZE) ** 2
        + ((centre_y - IMAGE_SIZE / 2) / IMAGE_SIZE) ** 2
    )

    centre_score = torch.clamp(1.0 - centre_distance * 2.0, 0.0, 1.0)

    return torch.clamp((size_score + centre_score) / 2.0, 0.0, 1.0)


def face_symmetry_score(landmarks):
    """
    Estimate facial symmetry using eyes, nose, and mouth landmarks.
    """
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    left_eye = landmarks[36:42].mean(dim=0)
    right_eye = landmarks[42:48].mean(dim=0)
    nose = landmarks[30]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2.0

    eye_width = torch.abs(right_eye[0] - left_eye[0]) + 1e-6
    mouth_width = torch.abs(right_mouth[0] - left_mouth[0]) + 1e-6

    nose_offset = torch.abs(nose[0] - eye_mid_x) / eye_width
    mouth_offset = torch.abs(mouth_mid_x - eye_mid_x) / mouth_width

    asymmetry = (nose_offset + mouth_offset) / 2.0

    return torch.clamp(1.0 - asymmetry, 0.0, 1.0).float()


def eye_mouth_artifact_score(landmarks):
    """
    Estimate possible eye and mouth shape artefacts.
    """
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    left_eye_width = torch.max(left_eye[:, 0]) - torch.min(left_eye[:, 0]) + 1e-6
    right_eye_width = torch.max(right_eye[:, 0]) - torch.min(right_eye[:, 0]) + 1e-6

    left_eye_height = torch.max(left_eye[:, 1]) - torch.min(left_eye[:, 1])
    right_eye_height = torch.max(right_eye[:, 1]) - torch.min(right_eye[:, 1])

    left_eye_ratio = left_eye_height / left_eye_width
    right_eye_ratio = right_eye_height / right_eye_width

    eye_mismatch = torch.abs(left_eye_ratio - right_eye_ratio)

    mouth_width = torch.max(mouth[:, 0]) - torch.min(mouth[:, 0]) + 1e-6
    mouth_height = torch.max(mouth[:, 1]) - torch.min(mouth[:, 1])
    mouth_ratio = mouth_height / mouth_width

    artifact_score = eye_mismatch + torch.abs(mouth_ratio - 0.35)

    return torch.clamp(artifact_score, 0.0, 1.0).float()


def texture_frequency_artifact_score(image_pil):
    """
    Estimate texture artefacts using Laplacian variance.
    """
    image = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    score = float(laplacian.var()) * 10.0

    return torch.tensor(max(0.0, min(score, 1.0)), dtype=torch.float32)


def compute_image_features(image_pil):
    """
    Extract the four handcrafted image features used with the CNN model.
    """
    landmarks = get_landmarks(image_pil)

    features = torch.stack(
        [
            face_landmark_quality(landmarks),
            face_symmetry_score(landmarks),
            eye_mouth_artifact_score(landmarks),
            texture_frequency_artifact_score(image_pil),
        ]
    )

    return torch.clamp(features, 0.0, 1.0).float()


class ImageDataset(Dataset):
    """
    Loads processed image files and returns image tensors, extra features, and labels.
    """

    def __init__(self, split):
        self.split = split
        self.train_transform, self.eval_transform = self._build_transforms()
        self.samples = self._load_samples()

        self.real_count = sum(1 for _, label in self.samples if label == 0)
        self.fake_count = sum(1 for _, label in self.samples if label == 1)

        print(f"{split}: {len(self.samples)} images")
        print(f"{split}/real after balance: {self.real_count}")
        print(f"{split}/fake after balance: {self.fake_count}")

    def _build_transforms(self):
        """
        Create training and evaluation transforms.
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

    def _collect_label_samples(self, label_name, label):
        """
        Collect image paths for one class label.
        """
        folder = DATA_DIR / self.split / label_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.webp"))
        )

        return [(file_path, label) for file_path in files]

    def _load_samples(self):
        """
        Load real and fake image paths for the selected split.
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

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image_path, label = random.choice(self.samples)
            image = Image.open(image_path).convert("RGB")

        transform = self.train_transform if self.split == "train" else self.eval_transform

        image_tensor = transform(image)
        extra_features = compute_image_features(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, extra_features, label_tensor


def build_model():
    """
    Create the EfficientNet-B0 image classification model.
    """
    return create_fake_detection_model(
        num_classes=2,
        pretrained=True,
        dropout=0.4,
        extra_feature_dim=EXTRA_FEATURE_DIM,
        use_extra_features=USE_EXTRA_FEATURES,
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

    for images, extra_features, labels in loop:
        images = images.to(DEVICE, non_blocking=True)
        extra_features = extra_features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(images, extra_features)
                loss = loss_fn(logits, labels)

            if training:
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

        predictions = logits.argmax(dim=1)

        batch_size = images.size(0)
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
        "model": "efficientnet_b0_rgb_4_static_image_features",
        "image_size": IMAGE_SIZE,
        "dropout": 0.4,
        "extra_feature_dim": EXTRA_FEATURE_DIM,
        "use_extra_features": USE_EXTRA_FEATURES,
        "extra_features": IMAGE_FEATURE_NAMES,
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
    Train, validate, test, and save the image detection model.
    """
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = ImageDataset("train")
    eval_dataset = ImageDataset("eval")
    test_dataset = ImageDataset("test")

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