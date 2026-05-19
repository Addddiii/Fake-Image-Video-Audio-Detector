# ============================================================
# Video Preprocessing Script
# Extracts 32 face-cropped frames from each video.
# Output is used by train_video.py.
# ============================================================

from pathlib import Path
import shutil

import cv2
import dlib
import numpy as np
from tqdm import tqdm
from skimage import transform as trans


# =========================
# PATHS
# =========================

RAW_DIR = Path(r"D:\Videos\raw")
OUTPUT_DIR = Path(r"D:\Videos\processed")

DLIB_TOOLS = Path(r"C:\Users\moeya\Downloads\Fake-Image-Video-Audio-Detector\backend\dlib_tools")
PREDICTOR_PATH = DLIB_TOOLS / "shape_predictor_81_face_landmarks.dat"


# =========================
# SETTINGS
# =========================

FRAMES_PER_VIDEO = 32
IMAGE_SIZE = 256

CLEAN_OUTPUT = False  # set True only if you want to delete old processed folder

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


# =========================
# DLIB SETUP
# =========================

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing dlib predictor file: {PREDICTOR_PATH}"
    )

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# =========================
# FACE CROPPING
# =========================

def get_alignment_keypoints(image_rgb, face):
    shape = face_predictor(image_rgb, face)

    left_eye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    right_eye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    left_mouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    right_mouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    return np.concatenate(
        [left_eye, right_eye, nose, left_mouth, right_mouth],
        axis=0,
    ).astype(np.float32)


def crop_face_dlib(frame_bgr, image_size=256):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())

    try:
        keypoints = get_alignment_keypoints(rgb, face)
    except Exception:
        return None

    target_size = [112, 112]

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)

    dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * image_size / target_size[0]
    dst[:, 1] = dst[:, 1] * image_size / target_size[1]

    scale = 1.3
    margin_rate = scale - 1

    x_margin = image_size * margin_rate / 2
    y_margin = image_size * margin_rate / 2

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= image_size / (image_size + 2 * x_margin)
    dst[:, 1] *= image_size / (image_size + 2 * y_margin)

    tform = trans.SimilarityTransform()
    success = tform.estimate(keypoints, dst)

    if not success:
        return None

    matrix = tform.params[0:2, :]

    cropped_rgb = cv2.warpAffine(
        rgb,
        matrix,
        (image_size, image_size),
    )

    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

    return cropped_bgr


def crop_face_haar(frame_bgr, image_size=256):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    margin = int(0.30 * max(w, h))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_bgr.shape[1], x + w + margin)
    y2 = min(frame_bgr.shape[0], y + h + margin)

    face = frame_bgr[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return cv2.resize(face, (image_size, image_size))


def crop_face(frame_bgr, image_size=256):
    face = crop_face_dlib(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    face = crop_face_haar(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    return cv2.resize(frame_bgr, (image_size, image_size))


# =========================
# FRAME EXTRACTION
# =========================

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(
        0,
        frame_count - 1,
        FRAMES_PER_VIDEO,
        endpoint=True,
        dtype=int,
    )

    faces = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = crop_face(frame, image_size=IMAGE_SIZE)
        faces.append(face)

    cap.release()

    if len(faces) == 0:
        return []

    while len(faces) < FRAMES_PER_VIDEO:
        faces.append(faces[-1].copy())

    return faces[:FRAMES_PER_VIDEO]


# =========================
# PROCESSING
# =========================

def collect_videos(split, label_name):
    folder = RAW_DIR / split / label_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    videos = []

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(file_path)

    return sorted(videos)


def save_video_frames(frames, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        output_path = output_folder / f"frame_{i:03d}.png"
        cv2.imwrite(str(output_path), frame)


def process_split_label(split, label_name):
    videos = collect_videos(split, label_name)

    print(f"\n{split}/{label_name}: found {len(videos)} videos")

    saved_count = 0
    skipped_count = 0

    for idx, video_path in enumerate(
        tqdm(videos, desc=f"Processing {split}/{label_name}", unit="video"),
        start=1,
    ):
        frames = extract_frames_from_video(video_path)

        if len(frames) == 0:
            skipped_count += 1
            continue

        output_name = f"{label_name}_{idx:05d}"
        output_folder = OUTPUT_DIR / split / label_name / output_name

        save_video_frames(frames, output_folder)

        saved_count += 1

    print(f"{split}/{label_name} saved: {saved_count}")
    print(f"{split}/{label_name} skipped: {skipped_count}")


def print_final_counts():
    print("\nFinal processed folder counts:")

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            folder = OUTPUT_DIR / split / label_name

            if folder.exists():
                count = len([p for p in folder.iterdir() if p.is_dir()])
            else:
                count = 0

            print(f"{split}/{label_name}: {count}")


def main():
    if CLEAN_OUTPUT and OUTPUT_DIR.exists():
        print(f"Removing old processed folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            process_split_label(split, label_name)

    print_final_counts()

    print("\nDONE")
    print(f"Processed videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()