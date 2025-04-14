from flask import Blueprint, request, jsonify, current_app
import pandas as pd
from app.model import train_model, predict_country, predict_all_countries, get_model_metrics
from app.config import Config
import os
from datetime import datetime
import tempfile
import logging

api = Blueprint('api', __name__)

TEST_TRAINING_DATA_PATH = None


def _read_training_data():
    data_file = (
        current_app.config.get('TEST_TRAINING_DATA_PATH', Config.DATA_PATH)
        if current_app.config.get('TESTING')
        else Config.DATA_PATH
    )
    current_app.logger.debug(f"Reading data from: {data_file}")
    dataframe = pd.read_csv(data_file)
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    return dataframe


def _send_error_response(error_msg, status):
    current_app.logger.error(error_msg)
    return jsonify({"status": "error", "message": error_msg}), status


@api.route('/logs', methods=['GET'])
def logs_endpoint():
    log_location = current_app.config.get('LOG_PATH', Config.LOG_PATH)
    try:
        if not os.path.exists(log_location):
            return _send_error_response(f"Log file not found at: {log_location}", 404)
        with open(log_location, 'r', encoding='utf-8') as log_file:
            logs = log_file.read()
        current_app.logger.info("Logs retrieved successfully")
        return jsonify({"status": "success", "logs": logs}), 200
    except Exception as err:
        return _send_error_response(f"Error reading logs: {str(err)}", 500)


@api.route('/metrics', methods=['GET'])
def metrics_endpoint():
    model_location = current_app.config.get('MODEL_PATH', Config.MODEL_PATH)
    try:
        data = _read_training_data()
        metrics_result = get_model_metrics(model_location, data)
        current_app.logger.info(
            f"Metrics: RMSE={metrics_result['rmse']}, Countries={metrics_result['num_countries']}"
        )
        return jsonify({"status": "success", "metrics": metrics_result}), 200
    except FileNotFoundError:
        return _send_error_response("Model not found", 404)
    except Exception as err:
        return _send_error_response(f"Metrics error: {str(err)}", 500)


@api.route('/predict/all', methods=['POST'])
def predict_all_endpoint():
    if not request.is_json:
        return _send_error_response("JSON required", 400)

    input_data = request.get_json()
    if 'date' not in input_data:
        return _send_error_response("Missing required field: date", 400)

    date_input = input_data['date']
    try:
        parsed_date = datetime.strptime(date_input, '%Y-%m-%d')
    except ValueError:
        return _send_error_response(f"Invalid date format: {date_input}", 400)

    try:
        data = _read_training_data()
        model_location = current_app.config.get(
            'MODEL_PATH', Config.MODEL_PATH)
        predictions_list, total = predict_all_countries(
            parsed_date, model_location, data)
        current_app.logger.info(
            f"Predicted for {len(predictions_list)} countries on {date_input}")
        return jsonify(
            {
                "status": "success",
                "date": date_input,
                "total-revenue": float(total),
                "predictions": predictions_list,
            }
        ), 200
    except FileNotFoundError:
        return _send_error_response("Model or data not found", 404)
    except Exception as err:
        return _send_error_response(f"Prediction error: {str(err)}", 500)


@api.route('/predict/country', methods=['POST'])
def predict_country_endpoint():
    if not request.is_json:
        return _send_error_response("JSON required", 400)

    input_data = request.get_json()
    if not all(key in input_data for key in ['country', 'date']):
        return _send_error_response("Missing required fields: country, date", 400)

    country_name = input_data['country']
    date_input = input_data['date']
    try:
        parsed_date = datetime.strptime(date_input, '%Y-%m-%d')
    except ValueError:
        return _send_error_response(f"Invalid date format: {date_input}", 400)

    try:
        data = _read_training_data()
        model_location = current_app.config.get(
            'MODEL_PATH', Config.MODEL_PATH)
        revenue = predict_country(
            country_name, parsed_date, model_location, data)
        current_app.logger.info(
            f"Revenue for {country_name} on {date_input}: {revenue}")
        return jsonify(
            {
                "status": "success",
                "country": country_name,
                "date": date_input,
                "revenue": float(revenue),
            }
        ), 200
    except FileNotFoundError:
        return _send_error_response("Model or data not found", 404)
    except ValueError as err:
        return _send_error_response(str(err), 404)
    except Exception as err:
        return _send_error_response(f"Unexpected error: {str(err)}", 500)


@api.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return _send_error_response("No file uploaded", 400)

    uploaded_file = request.files['file']
    if not uploaded_file.filename:
        return _send_error_response("Empty file uploaded", 400)

    try:
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell()
        if file_size == 0:
            return _send_error_response("Empty file", 400)
        if file_size > Config.MAX_FILE_SIZE:
            return _send_error_response(f"File too large: {file_size} bytes", 400)
        uploaded_file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            uploaded_file.save(temp.name)
            temp_path = temp.name

        current_app.logger.debug(f"Saved file to: {temp_path}")
        data = pd.read_csv(temp_path)
        model_location = current_app.config.get(
            'MODEL_PATH', Config.MODEL_PATH)
        rmse_value = train_model(data, model_location)
        current_app.logger.info(f"Training complete, RMSE: {rmse_value}")
        return jsonify({"status": "success", "rmse": rmse_value}), 200
    except ValueError as err:
        return _send_error_response(str(err), 400)
    except Exception as err:
        return _send_error_response(str(err), 500)
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as err:
                current_app.logger.warning(
                    f"Failed to delete {temp_path}: {str(err)}")
