class WhlDailyPredictionJob < ApplicationJob
  queue_as :default

  def perform(*args)
    puts "------------------------------------------------------------------"

    whl_api_service = WhlApiService.new

    # Get the list of games that are happening the next day
    next_games = whl_api_service.game_id_url(num_of_days_ahead=1, num_of_past_games=2)
    next_games = next_games.dig("SiteKit", "Scorebar") || []

    next_games.each do |game|
      home_team = game["HomeLongName"]
      away_team = game["VisitorLongName"]
      game_id = game["ID"]

      # If we have already predicted this game, skip it
      # This is to prevent duplicate predictions for the same game
      if WhlPredictionRecord.exists?(game_id: game_id)
        puts "Game ID #{game_id} already has predictions. Skipping."
        next
      end

      [ 5, 10, 15 ].each do |k|
        home_past_stats = WhlTeamStat.calc_rolling_average(home_team, k)
        # puts "home_past_stats #{home_past_stats}"
        away_past_stats = WhlTeamStat.calc_rolling_average(away_team, k)
        # puts "away_past_stats #{away_past_stats}"


        home_prediction_stats = WhlTeamStat.calc_last_k_avg_stats(home_team, k, 1)
        away_prediction_stats = WhlTeamStat.calc_last_k_avg_stats(away_team, k, 0)

        prediction_payload = {
          past_stats: {
            home_team: home_past_stats,
            away_team: away_past_stats
          },
          predict_game: {
            home_team: home_prediction_stats,
            away_team: away_prediction_stats
          },
          home_team: home_team,
          away_team: away_team
        }

        # Get the game predictions for the next games
        prediction = MlChlService.calc_and_get_prediction(prediction_payload)

        WhlPredictionRecord.create(
          game_id: game_id,
          k_value: k,
          home_prob: prediction["home_team_prob"],
          away_prob: prediction["away_team_prob"],
          correct: nil,
          home_team: home_team,
          away_team: away_team
        )

        # return prediction
      end
    end

    nil
  end
end
