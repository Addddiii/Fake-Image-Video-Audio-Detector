"""
Manual model evaluation script.
Samples 20 real and 20 fake files from each version2 test dataset.
"""

import csv
import random
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from sklearn.metrics import accuracy_score, precision_score, recall_score

from inference.audio import initialize_model as initialize_audio_model
from inference.audio import predict_audio
from inference.image import initialize_model as initialize_image_model
from inference.image import predict_image
from inference.video import load_video_detector


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Trying audioread instead.*")


SAMPLES_PER_CLASS = 20
OUTPUT_CSV = "evaluation_results.csv"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]

IMAGE_MODEL_PATH = r"models\image_model.pth"
AUDIO_MODEL_PATH = r"models\audio_model.pth"
VIDEO_MODEL_PATH = r"models\video_model.pth"


initialize_image_model(IMAGE_MODEL_PATH)
initialize_audio_model(AUDIO_MODEL_PATH)
video_detector = load_video_detector(VIDEO_MODEL_PATH)


def collect_files(folder, extensions, limit):
    files = []

    for extension in extensions:
        files.extend(folder.rglob(f"*{extension}"))

    random.shuffle(files)

    return files[:limit]


def normalise_prediction(result):
    if not isinstance(result, dict):
        return "error", 0

    prediction = (
        result.get("prediction")
        or result.get("label")
        or result.get("verdict")
        or result.get("class")
    )

    confidence = (
        result.get("confidence_percent")
        or result.get("confidence")
        or result.get("probability")
        or result.get("score")
        or 0
    )

    if prediction is None:
        return "error", confidence

    prediction = str(prediction).lower()

    if "fake" in prediction:
        return "fake", confidence

    if "real" in prediction or "authentic" in prediction:
        return "real", confidence

    return "error", confidence


def run_prediction(media_type, sample_path):
    if media_type == "image":
        return predict_image(str(sample_path))

    if media_type == "audio":
        return predict_audio(str(sample_path))

    if media_type == "video":
        return video_detector.predict(str(sample_path))

    raise ValueError(f"Unknown media type: {media_type}")


def test_media_type(media_type, test_path, extensions):
    rows = []

    print(f"\nTesting {media_type.upper()}")

    for actual_label in ["real", "fake"]:
        folder = test_path / actual_label

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        samples = collect_files(folder, extensions, SAMPLES_PER_CLASS)
        print(f"{actual_label}: {len(samples)} samples")

        for sample_path in samples:
            try:
                result = run_prediction(media_type, sample_path)
                predicted_label, confidence = normalise_prediction(result)
                error_message = ""

            except Exception as error:
                predicted_label = "error"
                confidence = 0
                error_message = str(error)

            rows.append(
                {
                    "media_type": media_type,
                    "file_path": str(sample_path),
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "pass": actual_label == predicted_label,
                    "error": error_message,
                }
            )

    return rows


def calculate_metrics(rows):
    valid_rows = [
        row
        for row in rows
        if row["predicted_label"] in ["real", "fake"]
    ]

    if not valid_rows:
        return None

    y_true = [row["actual_label"] for row in valid_rows]
    y_pred = [row["predicted_label"] for row in valid_rows]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="fake",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="fake",
            zero_division=0,
        ),
        "total": len(valid_rows),
        "correct": sum(
            1
            for row in valid_rows
            if row["actual_label"] == row["predicted_label"]
        ),
    }


def print_metrics(name, metrics):
    print(f"\n===== {name.upper()} RESULTS =====")

    if metrics is None:
        print("No valid predictions found.")
        return

    print(f"Samples: {metrics['total']}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")


def save_csv(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "media_type",
                "file_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "pass",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)

    datasets = [
        {
            "media_type": "image",
            "path": Path(r"E:\FakeDetection\raw_datasets\image\version2\test"),
            "extensions": IMAGE_EXTENSIONS,
        },
        {
            "media_type": "audio",
            "path": Path(r"E:\FakeDetection\raw_datasets\audio\version2\test"),
            "extensions": AUDIO_EXTENSIONS,
        },
        {
            "media_type": "video",
            "path": Path(r"E:\FakeDetection\raw_datasets\video\version2\test"),
            "extensions": VIDEO_EXTENSIONS,
        },
    ]

    all_rows = []

    for dataset in datasets:
        rows = test_media_type(
            dataset["media_type"],
            dataset["path"],
            dataset["extensions"],
        )

        all_rows.extend(rows)

        metrics = calculate_metrics(rows)
        print_metrics(dataset["media_type"], metrics)

    save_csv(all_rows)

    overall_metrics = calculate_metrics(all_rows)
    print_metrics("overall", overall_metrics)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()