import pytest

from inference.audio import predict_audio
from inference.image import predict_image
from inference.video import load_video_detector


def test_missing_image_file_raises_error():
    with pytest.raises(Exception):
        predict_image("missing_image.jpg")


def test_missing_audio_file_raises_error():
    with pytest.raises(Exception):
        predict_audio("missing_audio.wav")


def test_missing_video_file_raises_error():
    detector = load_video_detector()

    with pytest.raises(Exception):
        detector.predict("missing_video.mp4")