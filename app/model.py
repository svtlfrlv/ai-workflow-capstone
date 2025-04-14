import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import json


def _verify_file_exists(file_path: str) -> None:
    # Check if the specified file exists, raise error if not
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")


def _encode_country(country: str, training_data: pd.DataFrame) -> int:
    # Encode country name using LabelEncoder
    encoder = LabelEncoder().fit(training_data['country'])
    return encoder.transform([country])[0]


def load_model(model_path: str):
    _verify_file_exists(model_path)
    # Retrieve the trained model from disk
    retrieved_model = joblib.load(model_path)
    return retrieved_model


def get_model_metrics(model_path: str, training_data: pd.DataFrame) -> dict:
    # Ensure model file exists
    _verify_file_exists(model_path)

    # Fetch model training timestamp
    model_timestamp = os.path.getmtime(model_path)
    formatted_date = datetime.fromtimestamp(
        model_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    # Load metrics from associated JSON file
    metrics_file = os.path.splitext(model_path)[0] + '_metrics.json'
    _verify_file_exists(metrics_file)
    with open(metrics_file, 'r') as file_handle:
        metrics_data = json.load(file_handle)
        error_metric = metrics_data.get('rmse', 0.0)

    # Calculate unique countries in dataset
    country_count = len(
        training_data['country'].unique()) if not training_data.empty else 0

    return {
        'rmse': error_metric,
        'training_date': formatted_date,
        'num_countries': country_count
    }


def predict_country(country: str, date: datetime, model_path: str, training_data: pd.DataFrame) -> float:
    # Load the prediction model
    predictor = load_model(model_path)

    # Validate country presence in training data
    if country not in training_data['country'].unique():
        raise ValueError(f"Country not found: {country}")

    # Prepare input features
    encoded_country = _encode_country(country, training_data)

    # Extract latest revenue for the country
    country_records = training_data[training_data['country'] == country].sort_values(
        'date')
    previous_revenue = country_records['revenue'].iloc[-1] if not country_records.empty else 0.0

    # Format date-based features
    input_features = pd.DataFrame([{
        'country_encoded': encoded_country,
        'month': date.month,
        'year': date.year,
        'times_viewed': 0.0,  # Default as no input provided
        'revenue_lag1': previous_revenue
    }])

    # Perform prediction
    predicted_value = predictor.predict(input_features)[0]
    return predicted_value


def predict_all_countries(date: datetime, model_path: str, training_data: pd.DataFrame) -> tuple:
    # Initialize model and encoder
    predictor = load_model(model_path)
    all_countries = training_data['country'].unique()
    encoder = LabelEncoder().fit(training_data['country'])

    # Build input data for all countries
    feature_records = []
    for country in all_countries:
        encoded_country = encoder.transform([country])[0]
        country_records = training_data[training_data['country'] == country].sort_values(
            'date')
        previous_revenue = country_records['revenue'].iloc[-1] if not country_records.empty else 0.0
        feature_records.append({
            'country_encoded': encoded_country,
            'month': date.month,
            'year': date.year,
            'times_viewed': 0.0,
            'revenue_lag1': previous_revenue
        })

    # Predict revenues
    feature_df = pd.DataFrame(feature_records)
    predicted_revenues = predictor.predict(feature_df)

    # Format output
    country_predictions = [
        {'country': country, 'revenue': float(revenue)}
        for country, revenue in zip(all_countries, predicted_revenues)
    ]
    total_predicted = sum(pred['revenue'] for pred in country_predictions)

    return country_predictions, total_predicted


def train_model(data: pd.DataFrame, model_path: str) -> float:
    # Validate input data schema
    expected_columns = {'year', 'month', 'country',
                        'revenue', 'times_viewed', 'date'}
    if not expected_columns.issubset(data.columns):
        raise ValueError("Invalid CSV schema")

    # Prepare data copy and features
    dataset = data.copy()
    dataset['date'] = pd.to_datetime(dataset['date'])
    encoder = LabelEncoder()
    dataset['country_encoded'] = encoder.fit_transform(dataset['country'])
    dataset['revenue_lag1'] = dataset.groupby(
        'country')['revenue'].shift(1).fillna(0)

    # Define feature set
    feature_columns = ['country_encoded', 'month',
                       'year', 'times_viewed', 'revenue_lag1']
    inputs = dataset[feature_columns]
    target = dataset['revenue']

    # Train model
    regressor = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42)
    error_metric = 0.0
    if len(dataset) >= 2:
        train_inputs, val_inputs, train_target, val_target = train_test_split(
            inputs, target, test_size=0.2, random_state=42
        )
        regressor.fit(train_inputs, train_target)
        predictions = regressor.predict(val_inputs)
        error_metric = root_mean_squared_error(val_target, predictions)
    else:
        regressor.fit(inputs, target)

    # Save model and metrics
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(regressor, model_path)
        metrics_file = os.path.splitext(model_path)[0] + '_metrics.json'
        with open(metrics_file, 'w') as file_handle:
            json.dump({'rmse': error_metric}, file_handle)
    except OSError as error:
        raise OSError(f"Failed to save model: {str(error)}")

    return error_metric
