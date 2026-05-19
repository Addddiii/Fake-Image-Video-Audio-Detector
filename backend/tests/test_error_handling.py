import pytest
from inference.image import predict_image
from inference.audio import predict_audio
from inference.video import predict_video


def test_missing_image_file_raises_error():
    with pytest.raises(Exception):
        predict_image("missing_image.jpg")


def test_missing_audio_file_raises_error():
    with pytest.raises(Exception):
        predict_audio("missing_audio.wav")


def test_missing_video_file_raises_error():
    with pytest.raises(Exception):
        predict_video("missing_video.mp4")