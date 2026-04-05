import os
import cv2
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
RAW_ROOT = r"D:\FakeDetection\raw_datasets\video\faceforensics++"
OUT_ROOT = r"D:\FakeDetection\processed_datasets\video"

REAL_FOLDERS = ["original"]
FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]

IMG_SIZE = (224, 224)          # resize frames
MAX_FRAMES_PER_VIDEO = 20      # extract up to 20 frames evenly per video
TRAIN_RATIO = 0.7
EVAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
random.seed(SEED)

# =========================
# HELPERS
# =========================
def find_videos(folder):
    videos = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return videos

def split_list(items, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_ratio)
    eval_end = train_end + int(n * eval_ratio)

    train_items = items[:train_end]
    eval_items = items[train_end:eval_end]
    test_items = items[eval_end:]

    return train_items, eval_items, test_items

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path, output_dir, max_frames=20, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"No frames found in: {video_path}")
        return 0

    ensure_dir(output_dir)

    step = max(1, total_frames // max_frames)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame = cv2.resize(frame, img_size)
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

            if saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved

def process_class(video_paths, split_name, class_name):
    for video_path in video_paths:
        video_name = Path(video_path).stem
        parent_name = Path(video_path).parent.name.lower()

        if class_name == "fake":
            save_name = f"{parent_name}_{video_name}"
        else:
            save_name = video_name

        out_dir = os.path.join(OUT_ROOT, split_name, class_name, save_name)

        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"Skipping already processed: {video_name}")
            continue

        saved = extract_frames(
            video_path=video_path,
            output_dir=out_dir,
            max_frames=MAX_FRAMES_PER_VIDEO,
            img_size=IMG_SIZE,
        )
        print(f"[{split_name}] {class_name}: {video_name} -> {saved} frames")

# =========================
# MAIN
# =========================
def main():
    all_real = []
    all_fake = []

    for folder in REAL_FOLDERS:
        path = os.path.join(RAW_ROOT, folder)
        if os.path.exists(path):
            vids = find_videos(path)
            all_real.extend(vids)
            print(f"Found {len(vids)} real videos in {folder}")
        else:
            print(f"Missing folder: {path}")

    for folder in FAKE_FOLDERS:
        path = os.path.join(RAW_ROOT, folder)
        if os.path.exists(path):
            vids = find_videos(path)
            all_fake.extend(vids)
            print(f"Found {len(vids)} fake videos in {folder}")
        else:
            print(f"Missing folder: {path}")

    print(f"\nTotal real videos: {len(all_real)}")
    print(f"Total fake videos: {len(all_fake)}")

    real_train, real_eval, real_test = split_list(all_real, TRAIN_RATIO, EVAL_RATIO, TEST_RATIO)
    fake_train, fake_eval, fake_test = split_list(all_fake, TRAIN_RATIO, EVAL_RATIO, TEST_RATIO)

    for split in ["train", "eval", "test"]:
        for cls in ["real", "fake"]:
            ensure_dir(os.path.join(OUT_ROOT, split, cls))

    process_class(real_train, "train", "real")
    process_class(real_eval, "eval", "real")
    process_class(real_test, "test", "real")

    process_class(fake_train, "train", "fake")
    process_class(fake_eval, "eval", "fake")
    process_class(fake_test, "test", "fake")

    print("\nDone.")

if __name__ == "__main__":
    main()