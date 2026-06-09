"""
Preprocess video datasets for video model training.

This script loads raw videos, samples frames evenly, detects and crops faces,
and saves each processed video as a folder of face-cropped frames.
"""

from pathlib import Path
import shutil

import cv2
import dlib
import numpy as np
from skimage import transform as trans
from tqdm import tqdm

RAW_DIR = Path(r"D:\videos_raw")
OUTPUT_DIR = Path(r"D:\videos_processed")

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"

FRAMES_PER_VIDEO = 32
IMAGE_SIZE = 256
CLEAN_OUTPUT = False

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(f"Missing dlib predictor file: {PREDICTOR_PATH}")

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def get_alignment_keypoints(image_rgb, face):
    """
    Extract five facial keypoints used for face alignment.
    """
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


def crop_face_dlib(frame_bgr, image_size=IMAGE_SIZE):
    """
    Detect, align, and crop the largest face using dlib landmarks.
    """
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

    destination = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    destination[:, 0] += 8.0
    destination[:, 0] = destination[:, 0] * image_size / target_size[0]
    destination[:, 1] = destination[:, 1] * image_size / target_size[1]

    scale = 1.3
    margin_rate = scale - 1

    x_margin = image_size * margin_rate / 2
    y_margin = image_size * margin_rate / 2

    destination[:, 0] += x_margin
    destination[:, 1] += y_margin

    destination[:, 0] *= image_size / (image_size + 2 * x_margin)
    destination[:, 1] *= image_size / (image_size + 2 * y_margin)

    try:
        alignment_transform = trans.SimilarityTransform.from_estimate(
            keypoints,
            destination,
        )
    except Exception:
        return None

    if alignment_transform is None:
        return None

    matrix = alignment_transform.params[0:2, :]
    cropped_rgb = cv2.warpAffine(rgb, matrix, (image_size, image_size))

    return cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)


def crop_face_haar(frame_bgr, image_size=IMAGE_SIZE):
    """
    Fallback face crop using OpenCV Haar cascade if dlib detection fails.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    margin = int(0.30 * max(width, height))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_bgr.shape[1], x + width + margin)
    y2 = min(frame_bgr.shape[0], y + height + margin)

    face = frame_bgr[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return cv2.resize(face, (image_size, image_size))


def crop_face(frame_bgr, image_size=IMAGE_SIZE):
    """
    Crop a face using dlib first, then Haar fallback.
    If no face is found, resize the full frame.
    """
    face = crop_face_dlib(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    face = crop_face_haar(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    return cv2.resize(frame_bgr, (image_size, image_size))


def extract_frames_from_video(video_path):
    """
    Sample evenly spaced frames from one video and crop faces from each frame.
    """
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

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success or frame is None:
            continue

        frames.append(crop_face(frame, image_size=IMAGE_SIZE))

    cap.release()

    if not frames:
        return []

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1].copy())

    return frames[:FRAMES_PER_VIDEO]


def collect_videos(split, label_name):
    """
    Collect all supported video files for a dataset split and class label.
    """
    folder = RAW_DIR / split / label_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    video_files = [
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(video_files)


def save_video_frames(frames, output_folder):
    """
    Save extracted frames into a folder for one processed video.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        output_path = output_folder / f"frame_{index:03d}.png"
        cv2.imwrite(str(output_path), frame)


def process_split_label(split, label_name):
    """
    Process one dataset split and label, then save frame folders.
    """
    videos = collect_videos(split, label_name)

    print(f"\n{split}/{label_name}: found {len(videos)} videos")

    saved_count = 0
    skipped_count = 0

    for index, video_path in enumerate(
        tqdm(videos, desc=f"Processing {split}/{label_name}", unit="video"),
        start=1,
    ):
        frames = extract_frames_from_video(video_path)

        if not frames:
            skipped_count += 1
            continue

        output_name = f"{label_name}_{index:05d}"
        output_folder = OUTPUT_DIR / split / label_name / output_name

        save_video_frames(frames, output_folder)
        saved_count += 1

    print(f"{split}/{label_name} saved: {saved_count}")
    print(f"{split}/{label_name} skipped: {skipped_count}")


def count_processed_videos(folder):
    """
    Count processed video folders inside a split/label folder.
    """
    if not folder.exists():
        return 0

    return len([path for path in folder.iterdir() if path.is_dir()])


def print_final_counts():
    """
    Print final processed video counts for each split and class.
    """
    print("\nFinal processed folder counts:")

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            folder = OUTPUT_DIR / split / label_name
            count = count_processed_videos(folder)

            print(f"{split}/{label_name}: {count}")


def main():
    """
    Run the full video preprocessing pipeline.
    """
    if CLEAN_OUTPUT and OUTPUT_DIR.exists():
        print(f"Removing old processed folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            process_split_label(split, label_name)

    print_final_counts()
    print(f"\nProcessed videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()