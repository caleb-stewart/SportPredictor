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
      game = WhlGame.find_by(game_id: avg.game_id)
      next unless game
      home_team = WhlTeam.find_by(hockeytech_id: game.home_team_id)
      away_team = WhlTeam.find_by(hockeytech_id: game.away_team_id)

      # Get previous rolling averages for each team before this game
      home_team_avgs = WhlRollingAverage.where(whl_team_id: home_team.id)
        .joins("INNER JOIN whl_games ON whl_rolling_averages.game_id = whl_games.game_id")
  .where("whl_games.game_date < ?", game.game_date)
  .order("whl_games.game_date ASC")
        .limit(avg.k_value)
        .map { |ra| {
          target_goals: ra.goals_for_avg.to_f,
          opponent_goals: ra.goals_against_avg.to_f,
          target_ppp: ra.power_play_percentage_avg.to_f,
          opponent_ppp: ra.power_play_percentage_against_avg.to_f,
          target_sog: ra.shots_for_avg.to_f,
          opponent_sog: ra.shots_against_avg.to_f,
          target_fowp: ra.faceoff_win_percentage_avg.to_f,
          opponent_fowp: ra.faceoff_win_percentage_against_avg.to_f,
          home_away: 1,
          goals_diff: ra.goals_diff.to_f,
          ppp_diff: ra.ppp_diff.to_f,
          sog_diff: ra.sog_diff.to_f,
          fowp_diff: ra.fowp_diff.to_f,
          target_win: ra.target_win.to_i
        } }

      away_team_avgs = WhlRollingAverage.where(whl_team_id: away_team.id)
        .joins("INNER JOIN whl_games ON whl_rolling_averages.game_id = whl_games.game_id")
  .where("whl_games.game_date < ?", game.game_date)
  .order("whl_games.game_date ASC")
        .limit(avg.k_value)
        .map { |ra| {
          target_goals: ra.goals_for_avg.to_f,
          opponent_goals: ra.goals_against_avg.to_f,
          target_ppp: ra.power_play_percentage_avg.to_f,
          opponent_ppp: ra.power_play_percentage_against_avg.to_f,
          target_sog: ra.shots_for_avg.to_f,
          opponent_sog: ra.shots_against_avg.to_f,
          target_fowp: ra.faceoff_win_percentage_avg.to_f,
          opponent_fowp: ra.faceoff_win_percentage_against_avg.to_f,
          home_away: 0,
          goals_diff: ra.goals_diff.to_f,
          ppp_diff: ra.ppp_diff.to_f,
          sog_diff: ra.sog_diff.to_f,
          fowp_diff: ra.fowp_diff.to_f,
          target_win: ra.target_win.to_i
        } }

      # Stats for the game to predict
      predict_home_stats = {
        target_goals: avg.goals_for_avg.to_f,
        opponent_goals: avg.goals_against_avg.to_f,
        target_ppp: avg.power_play_percentage_avg.to_f,
        opponent_ppp: avg.power_play_percentage_against_avg.to_f,
        target_sog: avg.shots_for_avg.to_f,
        opponent_sog: avg.shots_against_avg.to_f,
        target_fowp: avg.faceoff_win_percentage_avg.to_f,
        opponent_fowp: avg.faceoff_win_percentage_against_avg.to_f,
        home_away: 1,
        goals_diff: avg.goals_diff.to_f,
        ppp_diff: avg.ppp_diff.to_f,
        sog_diff: avg.sog_diff.to_f,
        fowp_diff: avg.fowp_diff.to_f
      }
      predict_away_stats = {
        target_goals: avg.goals_for_avg.to_f,
        opponent_goals: avg.goals_against_avg.to_f,
        target_ppp: avg.power_play_percentage_avg.to_f,
        opponent_ppp: avg.power_play_percentage_against_avg.to_f,
        target_sog: avg.shots_for_avg.to_f,
        opponent_sog: avg.shots_against_avg.to_f,
        target_fowp: avg.faceoff_win_percentage_avg.to_f,
        opponent_fowp: avg.faceoff_win_percentage_against_avg.to_f,
        home_away: 0,
        goals_diff: avg.goals_diff.to_f,
        ppp_diff: avg.ppp_diff.to_f,
        sog_diff: avg.sog_diff.to_f,
        fowp_diff: avg.fowp_diff.to_f
      }

      payload = {
        past_stats: {
          home_team: home_team_avgs,
          away_team: away_team_avgs
        },
        predict_game: {
          home_team: predict_home_stats,
          away_team: predict_away_stats
        },
        home_team_name: home_team.name,
        away_team_name: away_team.name
      }

      begin
        result = MlChlService.calc_and_get_prediction(payload)
        home_prob = result["home_team_prob"]
        away_prob = result["away_team_prob"]
        home_team_id = home_team&.id
        away_team_id = away_team&.id
        predicted_winner = home_prob > away_prob ? home_team_id : away_team_id
        # Determine actual winner from game
        actual_winner_id = if game.home_goal_count > game.away_goal_count
          home_team_id
        elsif game.home_goal_count < game.away_goal_count
          away_team_id
        else
          nil
        end
        WhlPredictionRecord.find_or_initialize_by(game_id: avg.game_id, k_value: avg.k_value).update!(
          home_team_id: home_team_id,
          away_team_id: away_team_id,
          home_team_probability: home_prob,
          away_team_probability: away_prob,
          predicted_winner_id: predicted_winner,
          actual_winner_id: actual_winner_id,
          correct: (actual_winner_id.present? && predicted_winner == actual_winner_id),
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
