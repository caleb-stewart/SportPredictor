import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from whl_prediction_utils import train_team_model, calculate_team_prob
from whl_v2_features import FeatureContractError
from whl_v2_inference import (
    ModelNotAvailableError,
    PayloadContractError,
    load_active_model,
    predict_from_payload,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "model_store", "whl_v2")

_MODEL_BUNDLE = None
_MODEL_POINTER_MTIME = None


def _get_active_bundle():
    global _MODEL_BUNDLE, _MODEL_POINTER_MTIME
    pointer_path = os.path.join(MODEL_ROOT, "active_model.json")

    try:
        current_mtime = os.path.getmtime(pointer_path)
    except OSError:
        current_mtime = None

    if _MODEL_BUNDLE is None or current_mtime != _MODEL_POINTER_MTIME:
        _MODEL_BUNDLE = load_active_model(model_root=MODEL_ROOT)
        _MODEL_POINTER_MTIME = current_mtime
    return _MODEL_BUNDLE


@app.route('/')
def index():
    return "Hello, World!"

# Test this using CURL with this command:
# curl -X POST http://localhost:2718/whl/calc_winner -H "Content-Type: application/json" -d @PredictorFlask/Spo_vs_MedHat_test.json
# @Spo_vs_MedHat_test.json has all of the training data and the prediction data to test this endpoint
@app.route('/whl/calc_winner', methods=['POST'])
def whl_predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "features_by_k" in data:
        try:
            prediction = predict_from_payload(data, _get_active_bundle())
            return jsonify(prediction)
        except (ModelNotAvailableError, PayloadContractError, FeatureContractError) as exc:
            return jsonify({"error": str(exc)}), 400

    # Legacy v1 payload fallback.
    try:
        home_team_data = data['past_stats']['home_team']
        away_team_data = data['past_stats']['away_team']
        home_team_name = data['home_team_name']
        away_team_name = data['away_team_name']
        home_team_pred = [data['predict_game']['home_team']]
        away_team_pred = [data['predict_game']['away_team']]

        home_trained_models = train_team_model(home_team_data)
        away_trained_models = train_team_model(away_team_data)
        home_prob = calculate_team_prob(home_team_pred, home_trained_models)
        away_prob = calculate_team_prob(away_team_pred, away_trained_models)
    except KeyError as exc:
        return jsonify({"error": f"Legacy payload missing key: {exc}"}), 400

    total_prob = home_prob + away_prob
    normalized_home_prob = home_prob / total_prob
    normalized_away_prob = away_prob / total_prob

    return jsonify(
        {
            "home_team_prob": normalized_home_prob,
            "away_team_prob": normalized_away_prob,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "model_family": "legacy_per_request_ensemble",
            "model_version": "legacy-v1",
            "k_components": {},
        }
    )


@app.route('/whl/model_status', methods=['GET'])
def model_status():
    try:
        bundle = _get_active_bundle()
    except ModelNotAvailableError as exc:
        return jsonify({"active": False, "error": str(exc)}), 404

    return jsonify(
        {
            "active": True,
            "model_version": bundle.get("model_version"),
            "model_family": bundle.get("model_family"),
            "k_values": bundle.get("k_values"),
        }
    )

if __name__ == "__main__":
    # Port: Eulers constant (2.718) :p
    app.run(debug=True, port=2718)
