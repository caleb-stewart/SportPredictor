class WhlGame < ApplicationRecord
  def predict_winner_with_probabilities(k_value: nil, num_past_games: nil)
    payload = WhlPredictionFeatureService.build_payload(game: self)
    return nil unless payload

    result = MlChlService.calc_and_get_prediction(payload)
    home_prob = result["home_team_prob"]
    away_prob = result["away_team_prob"]
    predicted_winner = WhlTeam.find_by(hockeytech_id: result["predicted_winner_id"]) ||
      WhlTeam.find_by(hockeytech_id: (home_prob > away_prob ? home_team_id : away_team_id))

    {
      predicted_winner: predicted_winner,
      home_team: WhlTeam.find_by(hockeytech_id: home_team_id),
      away_team: WhlTeam.find_by(hockeytech_id: away_team_id),
      home_team_probability: home_prob,
      away_team_probability: away_prob,
      model_version: result["model_version"],
      model_family: result["model_family"],
      k_components: result["k_components"]
    }
  end
end
