import csv
import random
import traceback
import logging
import warnings
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================================
# CLEAN TERMINAL OUTPUT
# ==========================================================

logging.getLogger("inference.image").setLevel(logging.WARNING)
logging.getLogger("inference.audio").setLevel(logging.WARNING)
logging.getLogger("inference.video").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# SETTINGS
# ==========================================================

SAMPLES_PER_CLASS = 50
OUTPUT_CSV = "evaluation_results.csv"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
AUDIO_EXTS = [".wav", ".mp3", ".flac"]
VIDEO_EXTS = [".mp4", ".avi", ".mov"]

# ==========================================================
# IMPORT PREDICTION FUNCTIONS
# ==========================================================

from inference.image import predict_image, initialize_model
from inference.video import predict_video
from inference.audio import predict_audio

initialize_model(r"models\image_model.pth")

# ==========================================================
# HELPERS
# ==========================================================

def collect_files(folder, exts, limit):
    files = []

    for ext in exts:
        files.extend(folder.rglob(f"*{ext}"))

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


def run_single_prediction(media_type, sample_path):
    if media_type == "image":
        return predict_image(str(sample_path))

    if media_type == "audio":
        return predict_audio(str(sample_path))

    if media_type == "video":
        return predict_video(str(sample_path))

    raise ValueError(f"Unknown media type: {media_type}")


def test_media_type(media_type, test_path, exts):
    rows = []

    print(f"\nTesting {media_type.upper()}")

    for actual_label in ["real", "fake"]:
        folder = test_path / actual_label

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        samples = collect_files(folder, exts, SAMPLES_PER_CLASS)
        print(f"{actual_label}: {len(samples)} samples")

        for sample_path in samples:
            try:
                result = run_single_prediction(media_type, sample_path)
                predicted_label, confidence = normalise_prediction(result)
                error_message = ""

            except Exception as e:
                predicted_label = "error"
                confidence = 0
                error_message = str(e)

            rows.append({
                "media_type": media_type,
                "file_path": str(sample_path),
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "pass": actual_label == predicted_label,
                "error": error_message
            })

    return rows


def calculate_metrics(rows):
    valid_rows = [r for r in rows if r["predicted_label"] in ["real", "fake"]]

    if not valid_rows:
        return None

    y_true = [r["actual_label"] for r in valid_rows]
    y_pred = [r["predicted_label"] for r in valid_rows]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="fake", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="fake", zero_division=0),
        "total": len(valid_rows),
        "correct": sum(1 for r in valid_rows if r["actual_label"] == r["predicted_label"])
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
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "media_type",
                "file_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "pass",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================================
# MAIN
# ==========================================================

def main():
    random.seed(42)

    datasets = [
        {
            "media_type": "image",
            "path": Path(r"E:\FakeDetection\raw_datasets\image\test"),
            "exts": IMAGE_EXTS
        },
        {
            "media_type": "audio",
            "path": Path(r"E:\FakeDetection\raw_datasets\audio\test"),
            "exts": AUDIO_EXTS
        },
        {
            "media_type": "video",
            "path": Path(r"E:\FakeDetection\raw_datasets\video\version1\test"),
            "exts": VIDEO_EXTS
        }
    ]

    all_rows = []

    for dataset in datasets:
        rows = test_media_type(
            dataset["media_type"],
            dataset["path"],
            dataset["exts"]
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