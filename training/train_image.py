# ============================================================
# Image Training Script
# ============================================================

import os
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = r"D:\FakeDetection\processed_datasets\image"
MODEL_DIR = r"D:\FakeDetection\models\image"
BEST_PATH = os.path.join(MODEL_DIR, "image_best.pth")

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 5

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


# =========================
# TRANSFORMS
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =========================
# DATA
# =========================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
eval_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "eval"),  transform=eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

print("Train:", len(train_ds), "Eval:", len(eval_ds), "Test:", len(test_ds))


# =========================
# MODEL
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


# =========================
# EPOCH FUNCTION
# =========================
def run_epoch(loader, training=False, name="Train"):
    model.train() if training else model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=name, dynamic_ncols=True)

    for x, y in loop:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(DEVICE.type=="cuda")):
                out = model(x)
                loss = criterion(out, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = out.argmax(1)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (preds == y).sum().item()
        total += bs

        loop.set_postfix(
            loss=f"{loss_sum/total:.4f}",
            acc=f"{correct/total:.4f}"
        )

    return loss_sum/total, correct/total


# =========================
# TRAIN LOOP
# =========================
best_loss = float("inf")
best_weights = copy.deepcopy(model.state_dict())
wait = 0

for epoch in range(1, EPOCHS+1):
    print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

    train_loss, train_acc = run_epoch(train_loader, True, "Train")
    eval_loss, eval_acc   = run_epoch(eval_loader, False, "Eval")

    scheduler.step(eval_loss)

    print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Eval  Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

    if eval_loss < best_loss:
        best_loss = eval_loss
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, BEST_PATH)
        print("Saved best model")
        wait = 0
    else:
        wait += 1
        print(f"No improvement ({wait}/{PATIENCE})")

    if wait >= PATIENCE:
        print("Early stopping")
        break


# =========================
# TEST
# =========================
print("\n===== TEST =====")

model.load_state_dict(best_weights)
test_loss, test_acc = run_epoch(test_loader, False, "Test")

print("\nFINAL RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc: {test_acc:.4f}")
print(f"Saved: {BEST_PATH}")