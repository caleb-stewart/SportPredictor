# lib/tasks/predictor_integration.rake
namespace :predictor do
  desc "Call PredictorFlask for each whl_rolling_averages row and store results in whl_prediction_records"
  task run: :environment do
    require "httparty"
    require "dotenv/load"
    flask_url = ENV["PREDICTOR_FLASK_URL"]
    ks = [ 5, 10, 15 ]
    logger = Logger.new(STDOUT)

    WhlRollingAverage.find_each do |avg|
      payload = {
        goals_for_avg: avg.goals_for_avg,
        goals_against_avg: avg.goals_against_avg,
        shots_for_avg: avg.shots_for_avg,
        shots_against_avg: avg.shots_against_avg,
        power_play_percentage_avg: avg.power_play_percentage_avg,
        power_play_percentage_against_avg: avg.power_play_percentage_against_avg,
        faceoff_win_percentage_avg: avg.faceoff_win_percentage_avg,
        faceoff_win_percentage_against_avg: avg.faceoff_win_percentage_against_avg,
        home_away: avg.home_away,
        goals_diff: avg.goals_diff,
        ppp_diff: avg.ppp_diff,
        sog_diff: avg.sog_diff,
        fowp_diff: avg.fowp_diff
      }
      begin
        response = HTTParty.post("#{flask_url}/predict", body: payload.to_json, headers: { "Content-Type" => "application/json" })
        result = response.parsed_response
        home_prob = result["home_team_probability"]
        away_prob = result["away_team_probability"]
        predicted_winner = home_prob > away_prob ? avg.whl_team_id : nil # You may want to adjust this logic
        WhlPredictionRecord.find_or_initialize_by(game_id: avg.game_id, k_value: avg.k_value).update!(
          home_team_id: avg.whl_team_id, # You may want to set this to the actual home team
          away_team_id: nil, # You may want to set this to the actual away team
          home_team_probability: home_prob,
          away_team_probability: away_prob,
          predicted_winner_id: predicted_winner,
          prediction_date: Time.now
        )
        logger.info "Prediction stored for game #{avg.game_id}, k=#{avg.k_value}"
      rescue => e
        logger.error "Prediction failed for game #{avg.game_id}, k=#{avg.k_value}: #{e.message}"
      end
    end
    logger.info "Predictor integration complete."
  end
end
