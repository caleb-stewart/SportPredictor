class MlChlService
  include HTTParty
  base_uri(ENV.fetch("PREDICTOR_FLASK_URL", "http://localhost:2718"))

  def self.calc_and_get_prediction(payload)
    response = self.post(
      "/whl/calc_winner",
      body: payload.to_json,
      headers: { "Content-Type" => "application/json" }
    )

    unless response.success?
      raise StandardError, "PredictorFlask request failed with #{response.code}: #{response.body}"
    end

    {
      "home_team_prob" => response["home_team_prob"]&.to_f,
      "away_team_prob" => response["away_team_prob"]&.to_f,
      "predicted_winner_id" => response["predicted_winner_id"],
      "model_version" => response["model_version"],
      "model_family" => response["model_family"],
      "k_components" => response["k_components"] || {}
    }
  end
end
