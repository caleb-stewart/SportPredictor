class WhlGame < ApplicationRecord
  def predict_winner_with_probabilities(k_value:, num_past_games:)
    home_team = WhlTeam.find_by(hockeytech_id: home_team_id)
    away_team = WhlTeam.find_by(hockeytech_id: away_team_id)
    return nil unless home_team && away_team

    home_team_avgs = WhlRollingAverage.where(whl_team_id: home_team.id, k_value: k_value)
      .joins("INNER JOIN whl_games ON whl_rolling_averages.game_id = whl_games.game_id")
  .where("whl_games.game_date < ?", game_date)
  .order("whl_games.game_date ASC")
      .limit(num_past_games)
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

    away_team_avgs = WhlRollingAverage.where(whl_team_id: away_team.id, k_value: k_value)
      .joins("INNER JOIN whl_games ON whl_rolling_averages.game_id = whl_games.game_id")
  .where("whl_games.game_date < ?", game_date)
  .order("whl_games.game_date ASC")
      .limit(num_past_games)
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

    latest_home_avg = home_team_avgs.last
    latest_away_avg = away_team_avgs.last

    predict_home_stats = latest_home_avg ? {
      target_goals: latest_home_avg[:target_goals],
      opponent_goals: latest_home_avg[:opponent_goals],
      target_ppp: latest_home_avg[:target_ppp],
      opponent_ppp: latest_home_avg[:opponent_ppp],
      target_sog: latest_home_avg[:target_sog],
      opponent_sog: latest_home_avg[:opponent_sog],
      target_fowp: latest_home_avg[:target_fowp],
      opponent_fowp: latest_home_avg[:opponent_fowp],
      home_away: 1,
      goals_diff: latest_home_avg[:goals_diff],
      ppp_diff: latest_home_avg[:ppp_diff],
      sog_diff: latest_home_avg[:sog_diff],
      fowp_diff: latest_home_avg[:fowp_diff]
    } : nil

    predict_away_stats = latest_away_avg ? {
      target_goals: latest_away_avg[:target_goals],
      opponent_goals: latest_away_avg[:opponent_goals],
      target_ppp: latest_away_avg[:target_ppp],
      opponent_ppp: latest_away_avg[:opponent_ppp],
      target_sog: latest_away_avg[:target_sog],
      opponent_sog: latest_away_avg[:opponent_sog],
      target_fowp: latest_away_avg[:target_fowp],
      opponent_fowp: latest_away_avg[:opponent_fowp],
      home_away: 0,
      goals_diff: latest_away_avg[:goals_diff],
      ppp_diff: latest_away_avg[:ppp_diff],
      sog_diff: latest_away_avg[:sog_diff],
      fowp_diff: latest_away_avg[:fowp_diff]
    } : nil

    return nil unless predict_home_stats && predict_away_stats

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

    result = MlChlService.calc_and_get_prediction(payload)
    home_prob = result["home_team_prob"]
    away_prob = result["away_team_prob"]
    predicted_winner = home_prob > away_prob ? home_team : away_team

    {
      predicted_winner: predicted_winner,
      home_team: home_team,
      away_team: away_team,
      home_team_probability: home_prob,
      away_team_probability: away_prob
    }
  end
end
