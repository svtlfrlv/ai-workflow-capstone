import pytest
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from app.model import train_model, load_model, predict_country, predict_all_countries, get_model_metrics


@pytest.fixture
def sample_data():
    """
    Provide a sample dataset for testing.
    """
    return pd.DataFrame({
        'year': [2020, 2020],
        'month': [1, 2],
        'country': ['TestCountry', 'TestCountry'],
        'revenue': [100.0, 200.0],
        'times_viewed': [10.0, 20.0],
        'date': ['2020-01-01', '2020-02-01']
    })


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """
    Create a temporary CSV file from sample data.
    """
    file_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def trained_model(tmp_path, sample_data):
    """
    Train and save a model for testing.
    """
    model_file = tmp_path / 'sample_model.pkl'
    train_model(sample_data, str(model_file))
    return str(model_file)


def _validate_metrics_format(performance_stats):
    """
    Helper to verify metrics dictionary structure.
    """
    assert isinstance(performance_stats, dict), "Metrics must be a dictionary"
    assert 'rmse' in performance_stats, "RMSE key missing"
    assert isinstance(performance_stats['rmse'], float), "RMSE must be a float"
    assert performance_stats['rmse'] >= 0, "RMSE must be non-negative"
    assert 'training_date' in performance_stats, "Training date missing"
    assert 'num_countries' in performance_stats, "Country count missing"


def test_get_metrics_OK(trained_model, sample_data):
    """
    Test retrieval of model performance metrics.
    """
    performance_stats = get_model_metrics(trained_model, sample_data)
    _validate_metrics_format(performance_stats)

    # Verify training date format
    try:
        datetime.strptime(
            performance_stats['training_date'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pytest.fail("Training date has invalid format")

    # Check country count
    expected_countries = len(sample_data['country'].unique())
    assert performance_stats['num_countries'] == expected_countries, "Incorrect country count"


def test_get_metrics_missing_file(tmp_path):
    """
    Test error handling for missing model file in metrics retrieval.
    """
    nonexistent_file = tmp_path / 'missing_model.pkl'
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        get_model_metrics(str(nonexistent_file), pd.DataFrame())


def test_load_OK(trained_model):
    """
    Test loading a valid model file.
    """
    loaded_regressor = load_model(trained_model)
    assert isinstance(
        loaded_regressor, RandomForestRegressor), "Loaded model is not a RandomForestRegressor"


def test_predict_OK(trained_model):
    """
    Test model prediction after loading.
    """
    loaded_regressor = load_model(trained_model)
    test_input = pd.DataFrame({
        'country_encoded': [0],
        'month': [1],
        'year': [2020],
        'times_viewed': [10.0],
        'revenue_lag1': [100.0]
    })
    result = loaded_regressor.predict(test_input)
    assert len(result) == 1, "Prediction should return one value"
    assert isinstance(result[0], float), "Prediction must be a float"
    assert result[0] >= 0, "Prediction must be non-negative"


def test_load_invalid_file():
    """
    Test error handling for missing model file during load.
    """
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model("invalid_model.pkl")


def test_predict_country_OK(trained_model, sample_data):
    """
    Test revenue prediction for a valid country.
    """
    prediction_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    predicted_revenue = predict_country(
        "TestCountry", prediction_date, trained_model, sample_data)
    assert isinstance(predicted_revenue,
                      float), "Predicted revenue must be a float"
    assert predicted_revenue >= 0, "Predicted revenue must be non-negative"


def test_predict_country_invalid_country(trained_model, sample_data):
    """
    Test error handling for unknown country in prediction.
    """
    prediction_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    with pytest.raises(ValueError, match="Country not found"):
        predict_country("InvalidCountry", prediction_date,
                        trained_model, sample_data)


def test_predict_all_countries_OK(trained_model, sample_data):
    """
    Test revenue predictions for all countries.
    """
    prediction_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    country_forecasts, total_forecast = predict_all_countries(
        prediction_date, trained_model, sample_data)

    # Validate output structure
    assert isinstance(country_forecasts, list), "Forecasts must be a list"
    assert len(country_forecasts) == len(
        sample_data['country'].unique()), "Incorrect number of forecasts"
    assert isinstance(total_forecast, float), "Total forecast must be a float"
    assert total_forecast >= 0, "Total forecast must be non-negative"

    # Check individual predictions
    for forecast in country_forecasts:
        assert 'country' in forecast, "Country key missing"
        assert 'revenue' in forecast, "Revenue key missing"
        assert isinstance(forecast['revenue'],
                          float), "Revenue must be a float"
        assert forecast['revenue'] >= 0, "Revenue must be non-negative"
        assert forecast['country'] in sample_data['country'].unique(
        ), "Unknown country in forecast"

    # Verify total matches sum of predictions
    calculated_total = sum(forecast['revenue']
                           for forecast in country_forecasts)
    assert abs(total_forecast -
               calculated_total) < 1e-6, "Total forecast mismatch"


def test_train_OK(sample_data, tmp_path):
    """
    Test model training with valid data.
    """
    model_file = tmp_path / 'new_model.pkl'
    training_error = train_model(sample_data, str(model_file))

    assert os.path.exists(model_file), "Model file was not created"
    assert isinstance(training_error, float), "Training error must be a float"
    assert training_error >= 0, "Training error must be non-negative"

    loaded_regressor = joblib.load(model_file)
    assert isinstance(
        loaded_regressor, RandomForestRegressor), "Trained model is not a RandomForestRegressor"


def test_train_invalid_data(tmp_path):
    """
    Test error handling for invalid data schema during training.
    """
    invalid_dataset = pd.DataFrame({
        'incorrect_field': [1, 2]
    })
    model_file = tmp_path / 'failed_model.pkl'
    with pytest.raises(ValueError, match="Invalid CSV schema"):
        train_model(invalid_dataset, str(model_file))


def test_train_small_dataset(tmp_path):
    """
    Test model training with minimal data.
    """
    minimal_data = pd.DataFrame({
        'year': [2020],
        'month': [1],
        'country': ['TestCountry'],
        'revenue': [100.0],
        'times_viewed': [10.0],
        'date': ['2020-01-01']
    })
    model_file = tmp_path / 'minimal_model.pkl'
    training_error = train_model(minimal_data, str(model_file))

    assert os.path.exists(model_file), "Model file was not created"
    assert isinstance(training_error, float), "Training error must be a float"
    assert training_error >= 0, "Training error must be non-negative"
