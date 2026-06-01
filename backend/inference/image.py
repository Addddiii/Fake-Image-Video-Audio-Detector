"""
Image deepfake detection inference.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import dlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from architectures.image_model import create_fake_detection_model


BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = BASE_DIR / "dlib_tools" / "shape_predictor_81_face_landmarks.dat"

IMAGE_SIZE = 224
EXTRA_FEATURE_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

if not PREDICTOR_PATH.exists():
    raise FileNotFoundError(f"Missing dlib predictor file: {PREDICTOR_PATH}")

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

_model_instance = None


def to_percent(value):
    return round(float(value) * 100.0, 2)


def to_score(value):
    return round(float(value), 4)


def get_confidence_level(confidence):
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.70:
        return "Medium"
    return "Low"


def make_summary(label, confidence):
    confidence_percent = to_percent(confidence)

    if confidence < 0.70:
        return (
            f"The image shows possible signs of being {label.upper()}, "
            f"but confidence is low ({confidence_percent}%). "
            "Manual review is recommended."
        )

    if confidence < 0.90:
        return (
            f"The image is predicted as {label.upper()} with "
            f"{confidence_percent}% confidence. "
            "This is a medium-confidence result, so review is recommended."
        )

    return f"This image is predicted to be {label.upper()} with {confidence_percent}% confidence."


def format_feature_score(value):
    return round(float(value), 3)


def get_landmarks(image_pil):
    image = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    shape = face_predictor(gray, face)

    points = [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]
    return torch.tensor(points, dtype=torch.float32)


def face_landmark_quality(landmarks):
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    x_min = landmarks[:, 0].min()
    x_max = landmarks[:, 0].max()
    y_min = landmarks[:, 1].min()
    y_max = landmarks[:, 1].max()

    face_w = x_max - x_min
    face_h = y_max - y_min

    size_score = torch.clamp((face_w * face_h) / (IMAGE_SIZE * IMAGE_SIZE), 0.0, 1.0)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    center_dist = torch.sqrt(
        ((center_x - IMAGE_SIZE / 2) / IMAGE_SIZE) ** 2
        + ((center_y - IMAGE_SIZE / 2) / IMAGE_SIZE) ** 2
    )

    center_score = torch.clamp(1.0 - center_dist * 2.0, 0.0, 1.0)

    return torch.clamp((size_score + center_score) / 2.0, 0.0, 1.0)


def face_symmetry_score(landmarks):
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    left_eye = landmarks[36:42].mean(dim=0)
    right_eye = landmarks[42:48].mean(dim=0)
    nose = landmarks[30]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2.0

    eye_width = torch.abs(right_eye[0] - left_eye[0]) + 1e-6
    mouth_width = torch.abs(right_mouth[0] - left_mouth[0]) + 1e-6

    nose_offset = torch.abs(nose[0] - eye_mid_x) / eye_width
    mouth_offset = torch.abs(mouth_mid_x - eye_mid_x) / mouth_width

    asymmetry = (nose_offset + mouth_offset) / 2.0
    return torch.clamp(1.0 - asymmetry, 0.0, 1.0).float()


def eye_mouth_artifact_score(landmarks):
    if landmarks is None or landmarks.shape[0] < 68:
        return torch.tensor(0.0)

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    left_eye_w = torch.max(left_eye[:, 0]) - torch.min(left_eye[:, 0]) + 1e-6
    right_eye_w = torch.max(right_eye[:, 0]) - torch.min(right_eye[:, 0]) + 1e-6

    left_eye_h = torch.max(left_eye[:, 1]) - torch.min(left_eye[:, 1])
    right_eye_h = torch.max(right_eye[:, 1]) - torch.min(right_eye[:, 1])

    left_eye_ratio = left_eye_h / left_eye_w
    right_eye_ratio = right_eye_h / right_eye_w

    eye_mismatch = torch.abs(left_eye_ratio - right_eye_ratio)

    mouth_w = torch.max(mouth[:, 0]) - torch.min(mouth[:, 0]) + 1e-6
    mouth_h = torch.max(mouth[:, 1]) - torch.min(mouth[:, 1])
    mouth_ratio = mouth_h / mouth_w

    artifact = eye_mismatch + torch.abs(mouth_ratio - 0.35)
    return torch.clamp(artifact, 0.0, 1.0).float()


def texture_frequency_artifact_score(image_pil):
    image = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    score = float(lap.var()) * 10.0

    return torch.tensor(max(0.0, min(score, 1.0)), dtype=torch.float32)


def compute_image_features(image_pil):
    landmarks = get_landmarks(image_pil)

    features = torch.stack(
        [
            face_landmark_quality(landmarks),
            face_symmetry_score(landmarks),
            eye_mouth_artifact_score(landmarks),
            texture_frequency_artifact_score(image_pil),
        ]
    )

    return torch.clamp(features, 0.0, 1.0).float()


class FakeImageDetector:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = torch.device(device) if device else DEVICE
        self.class_names = ["real", "fake"]

        self.model = None
        self.transform = None
        self.image_size = IMAGE_SIZE
        self.use_extra_features = True

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.image_size = checkpoint.get("image_size", IMAGE_SIZE)
            dropout = checkpoint.get("dropout", 0.4)
            extra_feature_dim = checkpoint.get("extra_feature_dim", EXTRA_FEATURE_DIM)
            self.use_extra_features = checkpoint.get("use_extra_features", True)
        else:
            state_dict = checkpoint
            dropout = 0.4
            extra_feature_dim = EXTRA_FEATURE_DIM
            self.use_extra_features = True

        self.model = create_fake_detection_model(
            num_classes=2,
            pretrained=False,
            dropout=dropout,
            extra_feature_dim=extra_feature_dim,
            use_extra_features=self.use_extra_features,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        extra_features = compute_image_features(image).unsqueeze(0).to(self.device)

        return image_tensor, extra_features

    def predict(self, image_path: str) -> Dict:
        image_tensor, extra_features = self.preprocess_image(image_path)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.model(image_tensor, extra_features)
                probs = torch.softmax(logits, dim=1)[0]

        real_prob = float(probs[0].detach().cpu())
        fake_prob = float(probs[1].detach().cpu())

        pred_index = int(torch.argmax(probs).item())
        label = self.class_names[pred_index]

        confidence = max(real_prob, fake_prob)

        return {
            "prediction": label,
            "label": label,
            "verdict": label.upper(),
            "confidence": to_score(confidence),
            "confidence_percent": to_percent(confidence),
            "confidence_level": get_confidence_level(confidence),
            "real_probability": to_score(real_prob),
            "fake_probability": to_score(fake_prob),
            "real_percent": to_percent(real_prob),
            "fake_percent": to_percent(fake_prob),
            "probabilities": {
                "real": to_percent(real_prob),
                "fake": to_percent(fake_prob),
            },
            "summary": make_summary(label, confidence),
            "technical_details": {
                "image_size": self.image_size,
                "device": str(self.device),
                "extra_features_used": self.use_extra_features,
            },
            "image_features": {
                "face_landmark_quality": format_feature_score(extra_features[0, 0].detach().cpu()),
                "face_symmetry_score": format_feature_score(extra_features[0, 1].detach().cpu()),
                "eye_mouth_artifact_score": format_feature_score(extra_features[0, 2].detach().cpu()),
                "texture_frequency_artifact_score": format_feature_score(extra_features[0, 3].detach().cpu()),
            },
        }


def initialize_model(model_path: str):
    global _model_instance

    if not Path(model_path).exists():
        return False

    try:
        _model_instance = FakeImageDetector(model_path)
        return True
    except Exception:
        _model_instance = None
        return False


def get_model() -> Optional[FakeImageDetector]:
    return _model_instance


def predict_image(image_path: str) -> Dict:
    if _model_instance is None:
        raise RuntimeError("Image model is not loaded.")

    return _model_instance.predict(image_path)