namespace :predictor_v2 do
  def persist_prediction_rows_for_game(game:, result:, logger:)
    ks = [ 5, 10, 15 ]
    home_team = WhlTeam.find_by(hockeytech_id: game.home_team_id)
    away_team = WhlTeam.find_by(hockeytech_id: game.away_team_id)
    return unless home_team && away_team

    ensemble_home_prob = result["home_team_prob"].to_f
    ensemble_away_prob = result["away_team_prob"].to_f
    predicted_hockeytech_id = result["predicted_winner_id"]
    predicted_winner_team = WhlTeam.find_by(hockeytech_id: predicted_hockeytech_id)

    actual_winner_id = if game.home_goal_count && game.away_goal_count
      if game.home_goal_count > game.away_goal_count
        home_team.id
      elsif game.home_goal_count < game.away_goal_count
        away_team.id
      end
    end

    ks.each do |k|
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

    logger.info "Stored v2 prediction rows for game #{game.game_id}"
  end

  desc "Train PredictorFlask v2 model package and promote if gates pass"
  task train: :environment do
    logger = Logger.new(STDOUT)

    py_bin = ENV.fetch("PREDICTOR_PYTHON_BIN", "PredictorFlask/pf-venv/bin/python")
    train_script = "PredictorFlask/train_whl_v2.py"

    db_hash = ActiveRecord::Base.connection_db_config.configuration_hash

    env = {
      "PGHOST" => db_hash[:host].to_s,
      "PGPORT" => db_hash[:port].to_s,
      "PGDATABASE" => db_hash[:database].to_s,
      "PGUSER" => db_hash[:username].to_s,
      "PGPASSWORD" => db_hash[:password].to_s
    }

    command = [
      py_bin,
      train_script
    ]

    logger.info "Running training command: #{command.join(' ')}"
    success = system(env, *command)
    raise "Predictor v2 training failed" unless success
  end

  desc "Predict tomorrow's games using active PredictorFlask v2 model"
  task predict_upcoming: :environment do
    logger = Logger.new(STDOUT)
    target_date = ENV["DATE"].present? ? Date.parse(ENV["DATE"]) : Date.current + 1

    games = WhlGame.where(game_date: target_date).order(:game_id)
    logger.info "Found #{games.count} games for #{target_date}"

    games.each do |game|
      payload = WhlPredictionFeatureService.build_payload(game: game)
      unless payload
        logger.info "Skipping game #{game.game_id}: missing pregame features"
        next
      end

      begin
        result = MlChlService.calc_and_get_prediction(payload)
      rescue => e
        logger.error "Prediction request failed for game #{game.game_id}: #{e.message}"
        next
      end

      persist_prediction_rows_for_game(game: game, result: result, logger: logger)
    end
  end

  desc "Daily pipeline: refresh rolling averages, retrain model, and predict upcoming games"
  task daily_pipeline: :environment do
    logger = Logger.new(STDOUT)

    logger.info "Running rolling averages refresh"
    Rake::Task["whl_rolling_averages:calculate"].reenable
    Rake::Task["whl_rolling_averages:calculate"].invoke

    logger.info "Running model training"
    Rake::Task["predictor_v2:train"].reenable
    Rake::Task["predictor_v2:train"].invoke

    logger.info "Running upcoming game predictions"
    Rake::Task["predictor_v2:predict_upcoming"].reenable
    Rake::Task["predictor_v2:predict_upcoming"].invoke
  end
end
