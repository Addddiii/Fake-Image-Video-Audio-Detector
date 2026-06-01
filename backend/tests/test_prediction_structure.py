def test_prediction_result_structure_example():
    result = {
        "prediction": "real",
        "confidence": 0.95,
        "real_percent": 95.0,
        "fake_percent": 5.0,
    }

    assert isinstance(result, dict)
    assert result["prediction"] in ["real", "fake"]
    assert "confidence" in result
    assert "real_percent" in result
    assert "fake_percent" in result