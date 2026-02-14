class WhlPredictionFeatureService
  K_VALUES = [ 5, 10, 15 ].freeze

  def self.build_payload(game:, k_values: K_VALUES)
    home_team = WhlTeam.find_by(hockeytech_id: game.home_team_id)
    away_team = WhlTeam.find_by(hockeytech_id: game.away_team_id)
    return nil unless home_team && away_team

    features_by_k = {}

    k_values.each do |k|
      home_avg = rolling_average_for(game: game, team: home_team, k_value: k)
      away_avg = rolling_average_for(game: game, team: away_team, k_value: k)

      return nil unless home_avg && away_avg

      features_by_k[k.to_s] = {
        home: feature_map_from_avg(home_avg),
        away: feature_map_from_avg(away_avg)
      }
    end

    {
      game_id: game.game_id,
      game_date: game.game_date&.iso8601,
      home_team_id: game.home_team_id,
      away_team_id: game.away_team_id,
      home_team_name: home_team.name,
      away_team_name: away_team.name,
      features_by_k: features_by_k
    }
  end

  def self.rolling_average_for(game:, team:, k_value:)
    exact = WhlRollingAverage.find_by(
      game_id: game.game_id,
      whl_team_id: team.id,
      k_value: k_value
    )

    return exact if exact

    WhlRollingAverage.where(whl_team_id: team.id, k_value: k_value)
      .joins("INNER JOIN whl_games ON whl_rolling_averages.game_id = whl_games.game_id")
      .where("whl_games.game_date < ?", game.game_date)
      .order("whl_games.game_date DESC")
      .first
  end

  def self.feature_map_from_avg(rolling_avg)
    {
      goals_for_avg: rolling_avg.goals_for_avg.to_f,
      goals_against_avg: rolling_avg.goals_against_avg.to_f,
      shots_for_avg: rolling_avg.shots_for_avg.to_f,
      shots_against_avg: rolling_avg.shots_against_avg.to_f,
      power_play_percentage_avg: rolling_avg.power_play_percentage_avg.to_f,
      power_play_percentage_against_avg: rolling_avg.power_play_percentage_against_avg.to_f,
      faceoff_win_percentage_avg: rolling_avg.faceoff_win_percentage_avg.to_f,
      faceoff_win_percentage_against_avg: rolling_avg.faceoff_win_percentage_against_avg.to_f,
      goals_diff: rolling_avg.goals_diff.to_f,
      ppp_diff: rolling_avg.ppp_diff.to_f,
      sog_diff: rolling_avg.sog_diff.to_f,
      fowp_diff: rolling_avg.fowp_diff.to_f
    }
  end
end
