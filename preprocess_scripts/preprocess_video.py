import os
import cv2

# =========================
# PATHS (UPDATED)
# =========================
RAW_ROOT = r"D:\video\raw"
OUT_ROOT = r"D:\video\processed"

FRAMES_PER_VIDEO = 20
IMG_SIZE = (224, 224)

# =========================
# CREATE OUTPUT STRUCTURE
# =========================
for split in ["train", "eval", "test"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)

# =========================
# EXTRACT FRAMES FUNCTION
# =========================
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Cannot read video: {video_path}")
        return

    step = max(total_frames // FRAMES_PER_VIDEO, 1)

    frame_id = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            frame = cv2.resize(frame, IMG_SIZE)

            filename = f"frame_{saved+1:02d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)

            saved += 1
            if saved >= FRAMES_PER_VIDEO:
                break

        frame_id += 1

    cap.release()

# =========================
# PROCESS ALL VIDEOS
# =========================
for split in ["train", "eval", "test"]:
    for cls in ["real", "fake"]:

        src_folder = os.path.join(RAW_ROOT, split, cls)
        dst_folder = os.path.join(OUT_ROOT, split, cls)

        if not os.path.exists(src_folder):
            print(f"Missing folder: {src_folder}")
            continue

        videos = [f for f in os.listdir(src_folder) if f.endswith(".mp4")]

        print(f"\nProcessing {split}/{cls} - {len(videos)} videos")

        for i, video in enumerate(videos, start=1):
            video_path = os.path.join(src_folder, video)

            video_name = os.path.splitext(video)[0]
            out_path = os.path.join(dst_folder, video_name)

            os.makedirs(out_path, exist_ok=True)

            extract_frames(video_path, out_path)

            if i % 50 == 0 or i == len(videos):
                print(f"   Progress: {i}/{len(videos)}")

print("\nDONE EXTRACTING FRAMES")