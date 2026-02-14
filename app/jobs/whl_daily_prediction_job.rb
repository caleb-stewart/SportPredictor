class WhlDailyPredictionJob < ApplicationJob
  queue_as :default

  def perform(*_args)
    logger = Logger.new(STDOUT)
    whl_api_service = WhlApiService.new

    next_games = whl_api_service.game_id_url(1, 2)
    next_games = next_games.dig("SiteKit", "Scorebar") || []

    next_games.each do |game_data|
      game = upsert_upcoming_game(game_data)
      next unless game

      payload = WhlPredictionFeatureService.build_payload(game: game)
      unless payload
        logger.info "Skipping upcoming game #{game.game_id}: missing feature rows"
        next
      end

      begin
        result = MlChlService.calc_and_get_prediction(payload)
      rescue => e
        logger.error "Prediction call failed for upcoming game #{game.game_id}: #{e.message}"
        next
      end

      persist_prediction_rows(game: game, result: result)
      logger.info "Stored upcoming predictions for game #{game.game_id}"
    end

    nil
  end

  private

  def upsert_upcoming_game(game_data)
    game_id = game_data["ID"]&.to_i
    return nil unless game_id

    game = WhlGame.find_or_initialize_by(game_id: game_id)
    game.assign_attributes(
      season_id: game_data["SeasonID"],
      season_name: game_data["SeasonName"],
      game_date: safe_iso8601_to_date(game_data["GameDateISO8601"]),
      venue: game_data["venue_name"],
      status: game_data["GameStatusString"],
      home_team_id: game_data["HomeID"]&.to_i,
      away_team_id: game_data["VisitorID"]&.to_i,
      home_team: game_data["HomeLongName"],
      away_team: game_data["VisitorLongName"]
    )
    game.save!
    game
  end

  def persist_prediction_rows(game:, result:)
    home_team = WhlTeam.find_by(hockeytech_id: game.home_team_id)
    away_team = WhlTeam.find_by(hockeytech_id: game.away_team_id)
    return unless home_team && away_team

    ensemble_home_prob = result["home_team_prob"].to_f
    ensemble_away_prob = result["away_team_prob"].to_f
    predicted_hockeytech_id = result["predicted_winner_id"]
    predicted_winner_team = WhlTeam.find_by(hockeytech_id: predicted_hockeytech_id)

    [ 5, 10, 15 ].each do |k|
      component = (result["k_components"] || {})[k.to_s] || {}
      home_prob = component["home_team_prob"]&.to_f || ensemble_home_prob
      away_prob = component["away_team_prob"]&.to_f || ensemble_away_prob
      predicted_winner_id = predicted_winner_team&.id || (home_prob >= away_prob ? home_team.id : away_team.id)

      WhlPredictionRecord.find_or_initialize_by(game_id: game.game_id, k_value: k).update!(
        home_team_id: home_team.id,
        away_team_id: away_team.id,
        home_team_probability: home_prob,
        away_team_probability: away_prob,
        predicted_winner_id: predicted_winner_id,
        actual_winner_id: nil,
        correct: nil,
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
  end

  def safe_iso8601_to_date(value)
    return nil if value.blank?

    Date.iso8601(value)
  rescue ArgumentError
    nil
  end
end
