from pathlib import Path
import cv2
import dlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(r"D:\Videos\processed")
FEATURE_CACHE_PATH = DATA_DIR / "video_feature_cache.pt"

PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"


# =========================
# SETTINGS
# =========================

IMAGE_SIZE = 256
FRAMES_PER_VIDEO = 32
MIN_FRAMES_REQUIRED = 16

VIDEO_EXTS = {".png", ".jpg", ".jpeg"}


# =========================
# DLIB SETUP
# =========================

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing dlib predictor file: {PREDICTOR_PATH}"
    )

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))


# =========================
# TRANSFORMS
# =========================

raw_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================
# FRAME SELECTION
# =========================

def select_frames(frames):
    """
    Always returns exactly 32 frame paths.

    32+ frames: evenly sample 32
    16–31 frames: use available frames, repeat last frame
    """

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


# =========================
# LANDMARK HELPERS
# =========================

def get_landmarks_from_pil(image):
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
    vertical_1 = torch.dist(eye_points[1], eye_points[5])
    vertical_2 = torch.dist(eye_points[2], eye_points[4])
    horizontal = torch.dist(eye_points[0], eye_points[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def mouth_opening_ratio(mouth_points):
    left = mouth_points[0]
    right = mouth_points[6]
    top = mouth_points[3]
    bottom = mouth_points[9]

    vertical = torch.dist(top, bottom)
    horizontal = torch.dist(left, right)

    return vertical / (horizontal + 1e-6)


def compute_landmark_features(landmarks_list):
    valid = [lm for lm in landmarks_list if lm is not None]

    if len(valid) < 2:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    landmarks = torch.stack(valid)

    # 1. Mouth/lip movement
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

    # 2. Face motion consistency
    stable_indices = [30, 36, 45, 48, 54]

    if landmarks.shape[1] > max(stable_indices):
        stable_points = landmarks[:, stable_indices, :]

        motion = torch.norm(
            stable_points[1:] - stable_points[:-1],
            dim=2
        ).mean(dim=1)

        motion = motion / IMAGE_SIZE

        if motion.numel() >= 2:
            motion_std = motion.std()
            face_motion_consistency = 1.0 / (1.0 + 10.0 * motion_std)
        else:
            face_motion_consistency = torch.tensor(0.0)
    else:
        face_motion_consistency = torch.tensor(0.0)

    # 3. Eye/blink movement
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


def compute_features_for_video(frame_paths):
    selected_frames = select_frames(frame_paths)

    raw_images = []
    landmarks_list = []

    for frame_path in selected_frames:
        image = Image.open(frame_path).convert("RGB")

        landmarks_list.append(get_landmarks_from_pil(image))
        raw_images.append(raw_tf(image))

    raw_images = torch.stack(raw_images)

    landmark_features = compute_landmark_features(landmarks_list)
    artifact_feature = compute_artifact_feature(raw_images).view(1)

    features = torch.cat([landmark_features, artifact_feature], dim=0)

    return features.float()


# =========================
# MAIN
# =========================

def main():
    feature_cache = {}

    samples = []

    for split in ["train", "eval", "test"]:
        for label in ["real", "fake"]:
            folder = DATA_DIR / split / label

            if not folder.exists():
                print(f"Missing folder: {folder}")
                continue

            for video_folder in folder.iterdir():
                if not video_folder.is_dir():
                    continue

                frames = (
                    sorted(video_folder.glob("*.png")) +
                    sorted(video_folder.glob("*.jpg")) +
                    sorted(video_folder.glob("*.jpeg"))
                )

                if len(frames) >= MIN_FRAMES_REQUIRED:
                    key = f"{split}/{label}/{video_folder.name}"
                    samples.append((key, frames))

    print(f"Total videos to precompute: {len(samples)}")

    for key, frames in tqdm(samples, desc="Precomputing features", unit="video"):
        try:
            feature_cache[key] = compute_features_for_video(frames)
        except Exception as e:
            print(f"Failed {key}: {e}")
            feature_cache[key] = torch.zeros(4, dtype=torch.float32)

    torch.save(feature_cache, FEATURE_CACHE_PATH)

    print("\nDONE")
    print(f"Saved feature cache to: {FEATURE_CACHE_PATH}")
    print(f"Total cached videos: {len(feature_cache)}")


if __name__ == "__main__":
    main()