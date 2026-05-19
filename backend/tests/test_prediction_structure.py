from inference.image import initialize_model, predict_image
from inference.audio import predict_audio
from inference.video import predict_video

initialize_model(r"models\image_model.pth")


def test_image_prediction_structure():
    result = predict_image(
        r"E:\FakeDetection\raw_datasets\image\test\real\real_000001.jpg"
    )

    assert isinstance(result, dict)
    assert result["prediction"] in ["real", "fake"]
    assert "confidence" in result
    assert 0 <= float(result["confidence"]) <= 100


def test_audio_prediction_structure():
    result = predict_audio(
        r"E:\FakeDetection\raw_datasets\audio\test\real\real_06509.wav"
    )

    assert isinstance(result, dict)
    assert result["prediction"] in ["real", "fake"]
    assert "confidence" in result
    assert 0 <= float(result["confidence"]) <= 100


def test_video_prediction_structure():
    result = predict_video(
        r"E:\FakeDetection\raw_datasets\video\version1\test\real\real_1592.mp4"
    )

    assert isinstance(result, dict)
    assert result["prediction"] in ["real", "fake"]
    assert "confidence" in result
    assert 0 <= float(result["confidence"]) <= 100