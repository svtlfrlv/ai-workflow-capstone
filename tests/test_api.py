import pytest
import os
import json
from datetime import datetime
from flask import Flask
from app import create_app, init_logging
from werkzeug.datastructures import FileStorage
import logging


@pytest.fixture
def test_client(tmp_path):
    app = create_app()
    app.config.update(
        TESTING=True,
        MODEL_PATH=str(tmp_path / "test_model.pkl"),
        LOG_PATH=str(tmp_path / "test_app.log"),
    )
    init_logging(app)
    with app.test_client() as client:
        yield client
    app_logger = app.logger
    for handler in app_logger.handlers[:]:
        handler.flush()
        handler.close()
        app_logger.removeHandler(handler)
    logging.getLogger("revenue_predictor").handlers.clear()


@pytest.fixture
def valid_csv():
    return "tests/test_data/mock_train.csv"


@pytest.fixture
def invalid_csv():
    return "tests/test_data/invalid_train.csv"


class TestApiRoutes:
    def test_logs_OK(self, test_client, valid_csv, tmp_path):
        log_path = tmp_path / "test_app.log"
        with open(valid_csv, "rb") as file:
            response = test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
            assert response.status_code == 200
        assert log_path.exists()
        response = test_client.get("/logs")
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert "logs" in response.json
        assert isinstance(response.json["logs"], str)
        assert "Training complete" in response.json["logs"]

    def test_logs_empty(self, test_client, tmp_path):
        log_path = tmp_path / "test_app.log"
        log_path.write_text("", encoding="utf-8")
        response = test_client.get("/logs")
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert response.json["logs"] == ""

    def test_logs_no_file(self, test_client, tmp_path):
        log_path = tmp_path / "test_app.log"
        if log_path.exists():
            test_client.application.logger.handlers.clear()
            logging.getLogger("revenue_predictor").handlers.clear()
            os.remove(log_path)
        response = test_client.get("/logs")
        assert response.status_code == 404
        assert response.json["status"] == "error"
        assert "Log file not found" in response.json["message"]

    def test_metrics_OK(self, test_client, valid_csv):
        with open(valid_csv, "rb") as file:
            test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert "metrics" in response.json
        metrics = response.json["metrics"]
        assert isinstance(metrics["rmse"], float)
        assert metrics["rmse"] >= 0
        assert isinstance(metrics["num_countries"], int)
        assert metrics["num_countries"] > 0
        assert "training_date" in metrics
        datetime.strptime(metrics["training_date"], "%Y-%m-%d %H:%M:%S")

    def test_metrics_no_model(self, test_client, tmp_path):
        for path in [tmp_path / "test_model.pkl", tmp_path / "test_model_metrics.json"]:
            if path.exists():
                os.remove(path)
        for fname in ["test_model.pkl", "test_model_metrics.json"]:
            residual_path = os.path.join("tests/test_models", fname)
            if os.path.exists(residual_path):
                os.remove(residual_path)
        response = test_client.get("/metrics")
        assert response.status_code == 404
        assert response.json["status"] == "error"
        assert "Model not found" in response.json["message"]

    def test_predict_all_OK(self, test_client, valid_csv):
        with open(valid_csv, "rb") as file:
            test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
        test_client.application.config["TEST_TRAINING_DATA_PATH"] = valid_csv
        payload = {"date": "2020-01-01"}
        response = test_client.post(
            "/predict/all",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert response.json["date"] == "2020-01-01"
        assert isinstance(response.json["total-revenue"], float)
        assert response.json["total-revenue"] >= 0
        assert isinstance(response.json["predictions"], list)
        assert len(response.json["predictions"]) > 0
        total = sum(pred["revenue"] for pred in response.json["predictions"])
        assert abs(response.json["total-revenue"] - total) < 1e-6

    def test_predict_all_invalid_date(self, test_client):
        payload = {"date": "2020-13-01"}
        response = test_client.post(
            "/predict/all",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Invalid date format" in response.json["message"]

    def test_predict_all_missing_date(self, test_client):
        payload = {}
        response = test_client.post(
            "/predict/all",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Missing required field" in response.json["message"]

    def test_predict_country_OK(self, test_client, valid_csv):
        with open(valid_csv, "rb") as file:
            test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
        test_client.application.config["TEST_TRAINING_DATA_PATH"] = valid_csv
        payload = {"country": "TestCountry", "date": "2020-01-01"}
        response = test_client.post(
            "/predict/country",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert response.json["country"] == "TestCountry"
        assert response.json["date"] == "2020-01-01"
        assert isinstance(response.json["revenue"], float)
        assert response.json["revenue"] >= 0

    def test_predict_country_invalid_country(self, test_client, valid_csv):
        with open(valid_csv, "rb") as file:
            test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
        test_client.application.config["TEST_TRAINING_DATA_PATH"] = valid_csv
        payload = {"country": "UnknownCountry", "date": "2020-01-01"}
        response = test_client.post(
            "/predict/country",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 404
        assert response.json["status"] == "error"
        assert "Country not found" in response.json["message"]

    def test_predict_country_invalid_date(self, test_client):
        payload = {"country": "TestCountry", "date": "2020-13-01"}
        response = test_client.post(
            "/predict/country",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Invalid date format" in response.json["message"]

    def test_predict_country_missing_parameters(self, test_client):
        payload = {"country": "TestCountry"}
        response = test_client.post(
            "/predict/country",
            content_type="application/json",
            data=json.dumps(payload),
        )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Missing required fields" in response.json["message"]

    def test_train_OK(self, test_client, valid_csv, tmp_path):
        model_path = tmp_path / "test_model.pkl"
        with open(valid_csv, "rb") as file:
            response = test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "mock_train.csv")},
            )
        assert response.status_code == 200
        assert response.json["status"] == "success"
        assert "rmse" in response.json
        assert model_path.exists()

    def test_train_invalid_schema(self, test_client, invalid_csv):
        with open(invalid_csv, "rb") as file:
            response = test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "invalid_train.csv")},
            )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Invalid CSV schema" in response.json["message"]

    def test_train_empty_dataset_file(self, test_client):
        empty_file = FileStorage(stream=open(
            os.devnull, "rb"), filename="empty.csv")
        response = test_client.post(
            "/train",
            content_type="multipart/form-data",
            data={"file": empty_file},
        )
        assert response.status_code == 400
        assert response.json["status"] == "error"
        assert "Empty file" in response.json["message"]

    def test_train_large_file_OK(self, test_client, tmp_path):
        large_csv = tmp_path / "large.csv"
        with open(large_csv, "w") as file:
            file.write("year,month,country,revenue,times_viewed,date\n")
            for _ in range(10000):
                file.write("2020,1,TestCountry,100.0,10.0,2020-01-01\n")
        with open(large_csv, "rb") as file:
            response = test_client.post(
                "/train",
                content_type="multipart/form-data",
                data={"file": (file, "large.csv")},
            )
        assert response.status_code == 200
        assert response.json["status"] == "success"
