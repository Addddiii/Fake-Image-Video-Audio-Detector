from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "image" in data["models"]
    assert "video" in data["models"]
    assert "audio" in data["models"]


def test_upload_no_file_returns_error():
    response = client.post("/upload")

    assert response.status_code in [400, 422]