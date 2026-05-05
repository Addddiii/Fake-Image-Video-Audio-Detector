# ============================================================
# Video Frame Preprocessing Script
# ============================================================

import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = r"D:\Videos"
OUTPUT_ROOT = r"D:\Videos_Processed"

FRAMES_PER_VIDEO = 20
IMG_SIZE = 224
SEED = 42

SPLITS = ["train", "eval", "test"]
CLASSES = ["real", "fake"]

random.seed(SEED)


def get_jpeg_quality(split):
    if split == "train":
        return random.randint(60, 90)
    return 75


def center_crop(frame, crop_ratio=0.6):
    h, w = frame.shape[:2]

    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    x = (w - new_w) // 2
    y = (h - new_h) // 2

    return frame[y:y + new_h, x:x + new_w]


def augment_frame(frame):
    # brightness change
    if random.random() < 0.5:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=random.randint(-15, 15))

    # slight blur
    if random.random() < 0.3:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame


def extract_frames(video_path, output_dir, split):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    step = max(total_frames // FRAMES_PER_VIDEO, 1)

    saved = 0
    frame_index = 0
    target_frame = 0

    while cap.isOpened() and saved < FRAMES_PER_VIDEO:
        success, frame = cap.read()
        if not success:
            break

        if frame_index == target_frame:
            frame = center_crop(frame)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            if split == "train":
                frame = augment_frame(frame)

            output_path = os.path.join(output_dir, f"frame_{saved + 1:02d}.jpg")
            quality = get_jpeg_quality(split)

            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            saved += 1
            target_frame += step

        frame_index += 1

    cap.release()
    return saved > 0


def main():
    processed_count = 0
    bad_count = 0

    for split in SPLITS:
        for cls in CLASSES:
            input_dir = os.path.join(INPUT_ROOT, split, cls)
            output_dir = os.path.join(OUTPUT_ROOT, split, cls)

            Path(output_dir).mkdir(parents=True, exist_ok=True)

            videos = [
                file for file in os.listdir(input_dir)
                if file.lower().endswith(".mp4")
            ]

            print(f"\n=== {split.upper()} / {cls.upper()} ===")
            print(f"Videos: {len(videos)}")

            for video in tqdm(videos, desc=f"{split}-{cls}", unit="video", ncols=100):
                video_path = os.path.join(input_dir, video)
                video_name = os.path.splitext(video)[0]

                output_video_dir = os.path.join(output_dir, video_name)
                Path(output_video_dir).mkdir(parents=True, exist_ok=True)

                if extract_frames(video_path, output_video_dir, split):
                    processed_count += 1
                else:
                    bad_count += 1

    print("\n===== DONE =====")
    print(f"Processed videos: {processed_count}")
    print(f"Bad videos: {bad_count}")
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()