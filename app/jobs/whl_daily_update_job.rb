class WhlDailyUpdateJob < ApplicationJob
  queue_as :default

  # Updates the WHL team stats in the database with the last games played
  # Run this daily
  def perform(*args)

    puts "RUNNING WHL DAILY PREDICTION JOB"
    whl_api_service = WhlApiService.new

    # Get the list of games that were played yesterday
    # We really only need this for the game_id, so we can get the full team stats
    update_games = whl_api_service.game_id_url(num_of_days_ahead=0, num_of_past_games=1)
    update_games = update_games['SiteKit']['Scorebar']

    # Go through each game played yesterday
    update_games.each do |game|
      # Get the game_id from game played yesterday
      game_id = game["ID"].to_i

      # Skip if game already exists in database
      if WhlTeamStat.exists?(game_id: game_id)
        puts "Game ID #{game_id} already exists in database. Skipping."
        next
      end

      # Fetch full game stats
      # uses the game_id to get the full game stats
      game_stats = whl_api_service.get_game_stats_url(game_id)

      home_name = game_stats['GC']['Clock']['home_team']['name']
      away_name = game_stats['GC']['Clock']['visiting_team']['name']
      home_goals = game_stats['GC']['Clock']['home_goal_count'].to_i
      away_goals = game_stats['GC']['Clock']['visiting_goal_count'].to_i

      # Compute Power Play Percentage
      home_pp_total = game_stats['GC']['Clock']['power_play']['total']['home'].to_f
      # If there are no goals or no power plays, set goals to 0.0
      # This is a workaround for the API returning nil for pp goals
      home_pp_goals = game_stats['GC']['Clock']['power_play']['goals']['home']&.to_f || 0.0
      # if there are no power plays, set ppp to 0.0, else compute ppp
      home_ppp = home_pp_total > 0 ? (home_pp_goals / home_pp_total) : 0.0

      away_pp_total = game_stats["GC"]["Clock"]["power_play"]["total"]["visiting"].to_f
      away_pp_goals = game_stats["GC"]["Clock"]["power_play"]["goals"]["visiting"]&.to_f || 0.0
      away_ppp = away_pp_total > 0 ? (away_pp_goals / away_pp_total) : 0.0

      # Compute Shots on Goal
      home_sog = game_stats["GC"]["Clock"]["shots_on_goal"]["home"].values.map(&:to_i).sum
      away_sog = game_stats["GC"]["Clock"]["shots_on_goal"]["visiting"].values.map(&:to_i).sum

      # Compute Faceoff Win Percentage
      home_fow = game_stats["GC"]["Clock"]["fow"]["home"].to_f
      away_fow = game_stats["GC"]["Clock"]["fow"]["visiting"].to_f
      fow_total = home_fow + away_fow

      if fow_total > 0
        home_fowp = home_fow / fow_total
        away_fowp = away_fow / fow_total
      else
        home_fowp = 0.5
        away_fowp = 0.5
      end

      WhlTeamStat.create!(
        game_id: game_id,
        home_name: home_name,
        away_name: away_name,
        home_goals: home_goals,
        away_goals: away_goals,
        home_ppp: home_ppp,
        away_ppp: away_ppp,
        home_sog: home_sog,
        away_sog: away_sog,
        home_fowp: home_fowp,
        away_fowp: away_fowp
      )

      # Update the prediction records with the correct winner
      update_prediction_records(game_id, home_goals, away_goals)

    end
  end

  def update_prediction_records(game_id, home_goals, away_goals)
    # Determine actual winner (1 = home win, 0 = away win, nil = tie or unknown)
    actual_winner = home_goals > away_goals ? 1 : 0

    # Fetch and update all predictions for this game
    WhlPredictionRecord.where(game_id: game_id).find_each do |record|
      # predicted winner is the record with the greater prob
      predicted_winner = record.home_prob > record.away_prob ? 1 : 0
      # update if the predicted winner was the actual winner
      record.update(correct: predicted_winner == actual_winner ? 1 : 0)
    end
  end
end
