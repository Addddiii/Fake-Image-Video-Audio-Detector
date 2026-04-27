import os
import copy
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_frames=20,
        transform=None,
        train=False,
        balance_ratio=None,   # fake:real cap, e.g. 2 means at most 2 fake per 1 real
        seed=42
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.train = train
        self.balance_ratio = balance_ratio
        self.seed = seed

        self.class_to_idx = {"fake": 0, "real": 1}
        self.samples = []

        fake_samples = []
        real_samples = []

        for class_name in ["fake", "real"]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for video_folder in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_path):
                    frame_paths = sorted([
                        os.path.join(video_path, f)
                        for f in os.listdir(video_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])

                    if len(frame_paths) > 0:
                        sample = {
                            "video_path": video_path,
                            "frame_paths": frame_paths,
                            "label": self.class_to_idx[class_name]
                        }

                        if class_name == "fake":
                            fake_samples.append(sample)
                        else:
                            real_samples.append(sample)

        # cap fake count in train set
        if self.train and self.balance_ratio is not None and len(real_samples) > 0:
            rng = random.Random(self.seed)
            max_fake = min(len(fake_samples), len(real_samples) * self.balance_ratio)
            if len(fake_samples) > max_fake:
                fake_samples = rng.sample(fake_samples, max_fake)

        self.samples = fake_samples + real_samples

        if self.train:
            random.Random(self.seed).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, frame_paths):
        n = len(frame_paths)

        if n >= self.num_frames:
            if self.train:
                # random temporal sampling for robustness
                indices = sorted(random.sample(range(n), self.num_frames))
            else:
                # deterministic for eval/test
                indices = torch.linspace(0, n - 1, steps=self.num_frames)
                indices = indices.round().long().tolist()
        else:
            indices = list(range(n)) + [n - 1] * (self.num_frames - n)

        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = self._sample_frames(sample["frame_paths"])
        frames = []

        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # [T, C, H, W]
        label = sample["label"]
        return frames, label


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.backbone(x)          # [B*T, F]
        features = self.dropout(features)
        logits = self.classifier(features)   # [B*T, 2]

        logits = logits.view(b, t, -1)       # [B, T, 2]
        logits = logits.mean(dim=1)          # temporal average
        return logits


def main():
    seed_everything(42)

    # =========================
    # PATHS
    # =========================
    DATA_DIR = r"D:\video\processed"
    MODEL_DIR = r"D:\video\model"
    MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_video_model_best.pth")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # SETTINGS
    # =========================
    IMG_SIZE = 224
    NUM_FRAMES = 20
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    EARLY_STOPPING_PATIENCE = 6
    LABEL_SMOOTHING = 0.03

    # stronger balance than before
    TRAIN_FAKE_TO_REAL_RATIO = 2

    # class weights: fake=0, real=1
    # higher weight on real because it is minority and harder
    CLASS_WEIGHTS = [1.0, 2.0]

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
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASETS
    # =========================
    train_dataset = VideoFrameDataset(
        os.path.join(DATA_DIR, "train"),
        num_frames=NUM_FRAMES,
        transform=train_transform,
        train=True,
        balance_ratio=TRAIN_FAKE_TO_REAL_RATIO,
        seed=42
    )

    eval_dataset = VideoFrameDataset(
        os.path.join(DATA_DIR, "eval"),
        num_frames=NUM_FRAMES,
        transform=eval_test_transform,
        train=False,
        balance_ratio=None
    )

    test_dataset = VideoFrameDataset(
        os.path.join(DATA_DIR, "test"),
        num_frames=NUM_FRAMES,
        transform=eval_test_transform,
        train=False,
        balance_ratio=None
    )

    print("\n========== DATASET INFO ==========")
    print("Train videos:", len(train_dataset))
    print("Eval videos :", len(eval_dataset))
    print("Test videos :", len(test_dataset))

    def count_labels(dataset):
        counts = {0: 0, 1: 0}
        for s in dataset.samples:
            counts[s["label"]] += 1
        return counts

    train_counts = count_labels(train_dataset)
    eval_counts = count_labels(eval_dataset)
    test_counts = count_labels(test_dataset)

    print("Train counts:", {"fake": train_counts[0], "real": train_counts[1]})
    print("Eval counts :", {"fake": eval_counts[0], "real": eval_counts[1]})
    print("Test counts :", {"fake": test_counts[0], "real": test_counts[1]})

    # =========================
    # SAMPLER FOR TRAINING
    # =========================
    train_labels = [s["label"] for s in train_dataset.samples]
    class_sample_count = torch.tensor([
        sum(1 for x in train_labels if x == 0),
        sum(1 for x in train_labels if x == 1)
    ], dtype=torch.float)

    class_sample_weights = 1.0 / class_sample_count
    sample_weights = [class_sample_weights[label].item() for label in train_labels]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # =========================
    # DATALOADERS
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
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
    model = VideoClassifier(num_classes=2, dropout=0.3).to(DEVICE)

    # =========================
    # LOSS / OPTIMIZER / SCHEDULER
    # =========================
    class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # =========================
    # METRICS
    # =========================
    def compute_metrics(all_preds, all_labels):
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        total_acc = (all_preds == all_labels).float().mean().item()

        fake_mask = (all_labels == 0)
        real_mask = (all_labels == 1)

        fake_acc = (
            (all_preds[fake_mask] == all_labels[fake_mask]).float().mean().item()
            if fake_mask.sum() > 0 else 0.0
        )
        real_acc = (
            (all_preds[real_mask] == all_labels[real_mask]).float().mean().item()
            if real_mask.sum() > 0 else 0.0
        )

        balanced_acc = (fake_acc + real_acc) / 2.0
        return total_acc, fake_acc, real_acc, balanced_acc

    # =========================
    # EPOCH LOOP
    # =========================
    def run_epoch(loader, training=False):
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for videos, labels in loader:
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = outputs.argmax(dim=1)

            batch_size_now = labels.size(0)
            total_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        avg_loss = total_loss / total_samples
        total_acc, fake_acc, real_acc, balanced_acc = compute_metrics(all_preds, all_labels)
        return avg_loss, total_acc, fake_acc, real_acc, balanced_acc

    # =========================
    # TRAINING LOOP
    # =========================
    best_eval_bal_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n========== Epoch {epoch + 1}/{NUM_EPOCHS} ==========")
        print(f"Learning Rate: {current_lr:.8f}")

        train_loss, train_acc, train_fake_acc, train_real_acc, train_bal_acc = run_epoch(train_loader, training=True)
        eval_loss, eval_acc, eval_fake_acc, eval_real_acc, eval_bal_acc = run_epoch(eval_loader, training=False)

        scheduler.step(eval_bal_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal Acc: {train_bal_acc:.4f}")
        print(f"Train Fake Acc: {train_fake_acc:.4f} | Train Real Acc: {train_real_acc:.4f}")

        print(f"Eval  Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | Bal Acc: {eval_bal_acc:.4f}")
        print(f"Eval  Fake Acc: {eval_fake_acc:.4f} | Eval  Real Acc: {eval_real_acc:.4f}")

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (NUM_EPOCHS - epoch - 1)

        print(f"Epoch Time: {epoch_time / 60:.2f} min")
        print(f"Estimated Remaining: {remaining_time / 60:.2f} min")

        if eval_bal_acc > best_eval_bal_acc:
            best_eval_bal_acc = eval_bal_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            save_obj = {
                "model_state_dict": best_model_weights,
                "num_frames": NUM_FRAMES,
                "img_size": IMG_SIZE,
                "class_to_idx": {"fake": 0, "real": 1},
                "best_eval_bal_acc": best_eval_bal_acc
            }
            torch.save(save_obj, MODEL_PATH)

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

    test_loss, test_acc, test_fake_acc, test_real_acc, test_bal_acc = run_epoch(test_loader, training=False)

    total_time = time.time() - start_time

    print("\n========== FINAL RESULTS ==========")
    print(f"Best Eval Balanced Accuracy: {best_eval_bal_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Fake Accuracy: {test_fake_acc:.4f}")
    print(f"Test Real Accuracy: {test_real_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Total Training Time: {total_time / 60:.2f} min")
    print("Training complete.")


if __name__ == "__main__":
    main()