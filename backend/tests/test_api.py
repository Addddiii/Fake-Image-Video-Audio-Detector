from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "models" in data
    assert "image" in data["models"]
    assert "video" in data["models"]
    assert "audio" in data["models"]


def test_predict_image_without_file_returns_error():
    response = client.post("/predict/image")

    assert response.status_code in [400, 422]


def test_predict_video_without_file_returns_error():
    response = client.post("/predict/video")

    assert response.status_code in [400, 422]


def test_predict_audio_without_file_returns_error():
    response = client.post("/predict/audio")

    assert response.status_code in [400, 422]