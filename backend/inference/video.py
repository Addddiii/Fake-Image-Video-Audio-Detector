from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skimage import transform as trans

from architectures.video_model import VideoClassifierLSTM


# =========================
# PATHS
# =========================

MODEL_PATH = BASE_DIR / "models" / "video_model.pth"

DLIB_TOOLS = BASE_DIR / "dlib_tools"
PREDICTOR_81 = DLIB_TOOLS / "shape_predictor_81_face_landmarks.dat"


# =========================
# DEFAULTS
# =========================

DEFAULT_FRAMES_PER_VIDEO = 12
DEFAULT_IMAGE_SIZE = 256

LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.5
EXTRA_FEATURE_DIM = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DLIB SETUP
# =========================

if not PREDICTOR_81.exists():
    raise FileNotFoundError(
        f"Missing dlib predictor file: {PREDICTOR_81}\n"
        "Put shape_predictor_81_face_landmarks.dat inside backend/dlib_tools."
    )

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_81))


# OpenCV fallback if dlib face crop fails
haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# =========================
# MODEL LOADING
# =========================

def load_video_model(model_path=MODEL_PATH):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

        frames_per_video = 12
        image_size = checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)
        lstm_hidden = checkpoint.get("lstm_hidden", LSTM_HIDDEN)
        lstm_layers = checkpoint.get("lstm_layers", LSTM_LAYERS)
        dropout = checkpoint.get("dropout", DROPOUT)
        extra_feature_dim = checkpoint.get("extra_feature_dim", EXTRA_FEATURE_DIM)

    else:
        state_dict = checkpoint

        frames_per_video = DEFAULT_FRAMES_PER_VIDEO
        image_size = DEFAULT_IMAGE_SIZE
        lstm_hidden = LSTM_HIDDEN
        lstm_layers = LSTM_LAYERS
        dropout = DROPOUT
        extra_feature_dim = EXTRA_FEATURE_DIM

    model = VideoClassifierLSTM(
        num_classes=2,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        in_channels=6,
        extra_feature_dim=extra_feature_dim,
        use_extra_features=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, frames_per_video, image_size


# =========================
# FACE CROP / ALIGNMENT
# =========================

def get_keypts(image_rgb, face):
    shape = face_predictor(image_rgb, face)

    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def crop_face_dlib(frame_bgr, res=256):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())

    try:
        keypoints = get_keypts(rgb, face).astype(np.float32)
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

    dst[:, 0] = dst[:, 0] * res / target_size[0]
    dst[:, 1] = dst[:, 1] * res / target_size[1]

    scale = 1.3
    margin_rate = scale - 1

    x_margin = res * margin_rate / 2
    y_margin = res * margin_rate / 2

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= res / (res + 2 * x_margin)
    dst[:, 1] *= res / (res + 2 * y_margin)

    tform = trans.SimilarityTransform()
    success = tform.estimate(keypoints, dst)

    if not success:
        return None

    matrix = tform.params[0:2, :]

    cropped_rgb = cv2.warpAffine(rgb, matrix, (res, res))
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

    return cropped_bgr


def crop_face_haar(frame_bgr, res=256):
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

    return cv2.resize(face, (res, res))


def crop_face(frame_bgr, res=256):
    face = crop_face_dlib(frame_bgr, res=res)

    if face is not None:
        return face

    face = crop_face_haar(frame_bgr, res=res)

    if face is not None:
        return face

    # Final fallback: full frame resized
    return cv2.resize(frame_bgr, (res, res))


# =========================
# VIDEO FRAME EXTRACTION
# =========================

def sharpness_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_video_faces(video_path, frames_per_video, image_size):
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in video: {video_path}")

    candidate_count = min(frame_count, frames_per_video * 3)

    frame_indices = np.linspace(
        0,
        frame_count - 1,
        candidate_count,
        endpoint=True,
        dtype=int,
    )

    candidates = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = crop_face(frame, res=image_size)
        score = sharpness_score(face)

        candidates.append({
            "index": int(frame_index),
            "face": face,
            "score": score,
        })

    cap.release()

    if len(candidates) == 0:
        raise RuntimeError(f"No usable frames found in video: {video_path}")

    # Pick the sharpest frames, then sort them back into timeline order
    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    selected = candidates[:frames_per_video]

    selected = sorted(
        selected,
        key=lambda item: item["index"]
    )

    faces = [item["face"] for item in selected]

    while len(faces) < frames_per_video:
        faces.append(faces[-1].copy())

    return faces[:frames_per_video]


# =========================
# LANDMARK FEATURE HELPERS
# =========================

def get_landmarks_from_pil(image, image_size):
    image = image.resize((image_size, image_size))
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


def compute_landmark_features(landmarks_list, image_size):
    valid = [lm for lm in landmarks_list if lm is not None]

    if len(valid) < 2:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    landmarks = torch.stack(valid)

    # 1. Mouth/lip landmark movement
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

    # 2. Face landmark motion consistency
    stable_indices = [30, 36, 45, 48, 54]

    if landmarks.shape[1] > max(stable_indices):
        stable_points = landmarks[:, stable_indices, :]

        motion = torch.norm(
            stable_points[1:] - stable_points[:-1],
            dim=2
        ).mean(dim=1)

        motion = motion / image_size

        if motion.numel() >= 2:
            motion_std = motion.std()
            face_motion_consistency = 1.0 / (1.0 + 10.0 * motion_std)
        else:
            face_motion_consistency = torch.tensor(0.0)
    else:
        face_motion_consistency = torch.tensor(0.0)

    # 3. Eye/blink landmark movement
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
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    gray = gray.unsqueeze(0).unsqueeze(0)

    lap = torch.nn.functional.conv2d(
        gray,
        kernel,
        padding=1,
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


def compute_extra_features(raw_frames, landmarks_list, image_size):
    landmark_features = compute_landmark_features(landmarks_list, image_size)
    artifact_feature = compute_artifact_feature(raw_frames).view(1)

    features = torch.cat(
        [landmark_features, artifact_feature],
        dim=0
    )

    return features.float()


# =========================
# PREPARE MODEL INPUT
# =========================

def prepare_video_tensor(face_frames, image_size):
    model_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    raw_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    model_images = []
    raw_images = []
    landmarks_list = []

    for frame_bgr in face_frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        landmarks_list.append(
            get_landmarks_from_pil(image, image_size)
        )

        model_images.append(model_tf(image))
        raw_images.append(raw_tf(image))

    images = torch.stack(model_images)
    raw_images = torch.stack(raw_images)

    extra_features = compute_extra_features(
        raw_images,
        landmarks_list,
        image_size
    )

    motion = torch.zeros_like(images)
    motion[1:] = torch.abs(images[1:] - images[:-1])

    images = torch.cat([images, motion], dim=1)

    images = images.unsqueeze(0)
    extra_features = extra_features.unsqueeze(0)

    return images, extra_features


# =========================
# PRODUCT OUTPUT HELPERS
# =========================

def to_percent(value):
    """
    Convert probability 0..1 into percentage with 2 decimals.
    Example:
        0.9876 -> 98.76
    """
    return round(float(value) * 100.0, 2)


def to_score(value):
    """
    Keep raw probability clean for backend/frontend use.
    Example:
        0.987663924 -> 0.9877
    """
    return round(float(value), 4)


def get_confidence_level(confidence):
    """
    Human-friendly confidence level for product display.
    """

    if confidence >= 0.90:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    else:
        return "Low"


def make_summary(label, confidence):
    confidence_percent = to_percent(confidence)

    if confidence < 0.70:
        return (
            f"The video shows possible signs of being {label.upper()}, "
            f"but confidence is low ({confidence_percent}%). "
            "Manual review is recommended."
        )

    if confidence < 0.90:
        return (
            f"The video is predicted as {label.upper()} with "
            f"{confidence_percent}% confidence. "
            "This is a medium-confidence result, so review is recommended."
        )

    return (
        f"This video is predicted to be {label.upper()} "
        f"with {confidence_percent}% confidence."
    )


def format_feature_score(value):
    """
    Format extra feature scores nicely.
    Values are already clamped between 0 and 1.
    """
    return round(float(value), 3)


# =========================
# PREDICTION
# =========================

def predict_video(video_path, model_path=MODEL_PATH):
    model, frames_per_video, image_size = load_video_model(model_path)

    face_frames = extract_video_faces(
        video_path,
        frames_per_video=frames_per_video,
        image_size=image_size,
    )

    videos, extra_features = prepare_video_tensor(
        face_frames,
        image_size=image_size,
    )

    videos = videos.to(DEVICE)
    extra_features = extra_features.to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(
            "cuda",
            enabled=(DEVICE.type == "cuda"),
        ):
            spatial_logits, temporal_logits, _ = model(
                videos,
                extra_features,
            )

            # Training used both heads:
            # loss = temporal_loss + 0.5 * spatial_loss
            # So inference should combine them the same way.
            combined_logits = temporal_logits + 0.5 * spatial_logits

            temporal_probs = torch.softmax(temporal_logits, dim=1)[0]
            spatial_probs = torch.softmax(spatial_logits, dim=1)[0]
            probs = torch.softmax(combined_logits, dim=1)[0]

    real_prob = float(probs[0].detach().cpu())
    fake_prob = float(probs[1].detach().cpu())

    pred_index = int(torch.argmax(probs).item())
    label = "real" if pred_index == 0 else "fake"

    confidence = max(real_prob, fake_prob)
    confidence_level = get_confidence_level(confidence)

    result = {
        # Main product result
        "prediction": label,
        "label": label,
        "verdict": label.upper(),

        # Clean product display values
        "confidence": to_score(confidence),
        "confidence_percent": to_percent(confidence),
        "confidence_level": confidence_level,

        "real_probability": to_score(real_prob),
        "fake_probability": to_score(fake_prob),

        "real_percent": to_percent(real_prob),
        "fake_percent": to_percent(fake_prob),

        # Helpful display text for frontend
        "summary": make_summary(label, confidence),

        # Model breakdown
        "analysis": {
            "temporal": {
                "real_probability": to_score(temporal_probs[0]),
                "fake_probability": to_score(temporal_probs[1]),
                "real_percent": to_percent(temporal_probs[0]),
                "fake_percent": to_percent(temporal_probs[1]),
            },
            "spatial": {
                "real_probability": to_score(spatial_probs[0]),
                "fake_probability": to_score(spatial_probs[1]),
                "real_percent": to_percent(spatial_probs[0]),
                "fake_percent": to_percent(spatial_probs[1]),
            },
        },

        # Technical details
        "technical_details": {
            "frames_used": frames_per_video,
            "image_size": image_size,
            "device": str(DEVICE),
        },

        # 4 extra handcrafted video features
        "video_features": {
            "mouth_lip_movement": format_feature_score(extra_features[0, 0].detach().cpu()),
            "face_motion_consistency": format_feature_score(extra_features[0, 1].detach().cpu()),
            "eye_blink_movement": format_feature_score(extra_features[0, 2].detach().cpu()),
            "artifact_compression_inconsistency": format_feature_score(extra_features[0, 3].detach().cpu()),
        },
    }

    return result


# =========================
# BACKEND COMPATIBILITY WRAPPER
# =========================

class VideoDetector:
    """
    Wrapper class used by main.py.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path

    def predict(self, video_path):
        return predict_video(
            video_path=video_path,
            model_path=self.model_path,
        )

    def analyze(self, video_path):
        return self.predict(video_path)


def load_video_detector(model_path=MODEL_PATH):
    return VideoDetector(model_path=model_path)
