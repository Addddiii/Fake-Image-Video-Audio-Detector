import pytest

from inference.audio import load_audio_detector
from inference.image import load_image_detector
from inference.video import load_video_detector


def test_missing_image_file_raises_error():
    """
    Check that image prediction raises an error for a missing file.
    """
    detector = load_image_detector()

    with pytest.raises(Exception):
        detector.predict("missing_image.jpg")


def test_missing_audio_file_raises_error():
    """
    Check that audio prediction raises an error for a missing file.
    """
    detector = load_audio_detector()

    with pytest.raises(Exception):
        detector.predict("missing_audio.wav")


def test_missing_video_file_raises_error():
    """
    Check that video prediction raises an error for a missing file.
    """
    detector = load_video_detector()

    with pytest.raises(Exception):
        detector.predict("missing_video.mp4")