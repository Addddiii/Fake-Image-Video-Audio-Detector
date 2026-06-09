"""
Preprocess image datasets for image model training.

This script loads raw images, detects and crops faces, resizes them to
224x224, and saves the processed images for model training.
"""

from pathlib import Path
import shutil

import cv2
import dlib
import numpy as np
from skimage import transform as trans
from tqdm import tqdm

RAW_DIR = Path(r"D:\images_raw")
OUTPUT_DIR = Path(r"D:\images_processed")

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"

IMAGE_SIZE = 224
CLEAN_OUTPUT = False

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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
    If no face is found, resize the full image.
    """
    face = crop_face_dlib(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    face = crop_face_haar(frame_bgr, image_size=image_size)

    if face is not None:
        return face

    return cv2.resize(frame_bgr, (image_size, image_size))


def collect_images(split, label_name):
    """
    Collect all supported image files for a dataset split and class label.
    """
    folder = RAW_DIR / split / label_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    image_files = [
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(image_files)


def process_split_label(split, label_name):
    """
    Process one dataset split and label, then save cropped face images.
    """
    images = collect_images(split, label_name)
    output_folder = OUTPUT_DIR / split / label_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{split}/{label_name}: found {len(images)} images")

    saved_count = 0
    skipped_count = 0

    for image_path in tqdm(
        images,
        desc=f"Processing {split}/{label_name}",
        unit="image",
    ):
        frame = cv2.imread(str(image_path))

        if frame is None:
            skipped_count += 1
            continue

        processed_image = crop_face(frame, image_size=IMAGE_SIZE)
        output_path = output_folder / image_path.name

        cv2.imwrite(str(output_path), processed_image)
        saved_count += 1

    print(f"{split}/{label_name} saved: {saved_count}")
    print(f"{split}/{label_name} skipped: {skipped_count}")


def count_processed_files(folder):
    """
    Count processed files inside a folder.
    """
    if not folder.exists():
        return 0

    return len([path for path in folder.iterdir() if path.is_file()])


def print_final_counts():
    """
    Print final processed image counts for each split and class.
    """
    print("\nFinal processed folder counts:")

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            folder = OUTPUT_DIR / split / label_name
            count = count_processed_files(folder)

            print(f"{split}/{label_name}: {count}")


def main():
    """
    Run the full image preprocessing pipeline.
    """
    if CLEAN_OUTPUT and OUTPUT_DIR.exists():
        print(f"Removing old processed folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            process_split_label(split, label_name)

    print_final_counts()
    print(f"\nProcessed images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()