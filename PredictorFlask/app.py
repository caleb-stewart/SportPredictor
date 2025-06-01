from flask import Flask, request, jsonify
from flask_cors import CORS

from whl_prediction_utils import train_team_model, calculate_team_prob

app = Flask(__name__)
CORS(app)

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
    
    # Get the past stats for home and away teams from POST request body
    home_team_data = data['past_stats']['home_team']
    # print("home_team_data head:", home_team_data)
    away_team_data = data['past_stats']['away_team']

    home_team_name = data['home_team']
    away_team_name = data['away_team']

    # Get the predicted game data for home and away teams from POST request body
    # Make sure this is a list of one dictionary
    home_team_pred = [data['predict_game']['home_team']]
    away_team_pred = [data['predict_game']['away_team']]

    # Train the models for the home and away teams
    home_trained_models = train_team_model(home_team_data)
    away_trained_models = train_team_model(away_team_data)

    # Calculate the probabilities of winning for home and away teams
    home_prob = calculate_team_prob(home_team_pred, home_trained_models)
    away_prob = calculate_team_prob(away_team_pred, away_trained_models)

    # Get the total probability of winning for both teams
    # We want the probabilities to sum to 1, but this may not always be the case
    # due to the nature of the classifiers and the data
    total_prob = home_prob + away_prob

    # normalize the probabilities to sum to 1
    normalized_home_prob = home_prob / total_prob
    normalized_away_prob = away_prob / total_prob

    return jsonify({"home_team_prob": normalized_home_prob,
                     "away_team_prob": normalized_away_prob,
                     "home_team": home_team_name,
                     "away_team": away_team_name})

if __name__ == "__main__":
    # Port: Eulers constant (2.718) :p
    app.run(debug=True, port=2718)