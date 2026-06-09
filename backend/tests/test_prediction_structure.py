def test_prediction_result_structure_example():
    """
    Check that a prediction result contains the expected response fields.
    """
    result = {
        "prediction": "real",
        "confidence": 0.95,
        "real_percent": 95.0,
        "fake_percent": 5.0,
    }

    assert isinstance(result, dict)
    assert result["prediction"] in ["real", "fake"]
    assert isinstance(result["confidence"], float)
    assert isinstance(result["real_percent"], float)
    assert isinstance(result["fake_percent"], float)