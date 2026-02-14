namespace :predictor do
  desc "Call PredictorFlask v2 for each game and store results in whl_prediction_records"
  task run: :environment do
    logger = Logger.new(STDOUT)
    ks = [ 5, 10, 15 ]

    WhlGame.order(:game_date, :game_id).find_each do |game|
      home_team = WhlTeam.find_by(hockeytech_id: game.home_team_id)
      away_team = WhlTeam.find_by(hockeytech_id: game.away_team_id)
      next unless home_team && away_team

      payload = WhlPredictionFeatureService.build_payload(game: game, k_values: ks)
      unless payload
        logger.info "Skipping game #{game.game_id}: missing features for one or more k values"
        next
      end

      begin
        result = MlChlService.calc_and_get_prediction(payload)
      rescue => e
        logger.error "Prediction failed for game #{game.game_id}: #{e.message}"
        next
      end

      ensemble_home_prob = result["home_team_prob"].to_f
      ensemble_away_prob = result["away_team_prob"].to_f
      predicted_winner_hockeytech_id = result["predicted_winner_id"]
      predicted_winner_team = WhlTeam.find_by(hockeytech_id: predicted_winner_hockeytech_id)

      actual_winner_id = if game.home_goal_count && game.away_goal_count
        if game.home_goal_count > game.away_goal_count
          home_team.id
        elsif game.home_goal_count < game.away_goal_count
          away_team.id
        end
      end

      ks.each do |k|
        component = (result["k_components"] || {})[k.to_s] || {}
        component_home_prob = component["home_team_prob"]&.to_f
        component_away_prob = component["away_team_prob"]&.to_f

        home_prob = component_home_prob || ensemble_home_prob
        away_prob = component_away_prob || ensemble_away_prob
        predicted_winner_id = predicted_winner_team&.id || (home_prob >= away_prob ? home_team.id : away_team.id)

        WhlPredictionRecord.find_or_initialize_by(game_id: game.game_id, k_value: k).update!(
          home_team_id: home_team.id,
          away_team_id: away_team.id,
          home_team_probability: home_prob,
          away_team_probability: away_prob,
          predicted_winner_id: predicted_winner_id,
          actual_winner_id: actual_winner_id,
          correct: actual_winner_id.present? ? (predicted_winner_id == actual_winner_id) : nil,
          prediction_date: Time.current,
          model_version: result["model_version"],
          model_family: result["model_family"],
          raw_model_outputs: {
            ensemble: {
              home_team_prob: ensemble_home_prob,
              away_team_prob: ensemble_away_prob
            },
            k_components: result["k_components"] || {}
          }
        )
      end

      logger.info "Prediction stored for game #{game.game_id}"
    end

    logger.info "Predictor integration complete."
  end
end
