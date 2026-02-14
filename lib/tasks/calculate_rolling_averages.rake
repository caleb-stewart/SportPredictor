# lib/tasks/calculate_rolling_averages.rake
namespace :whl_rolling_averages do
  desc "Calculate and store rolling averages for each team/game/k using whl_games data"
  task calculate: :environment do
    require "logger"
    logger = Logger.new(STDOUT)
    ks = [ 5, 10, 15 ]
    WhlTeam.find_each do |team|
      logger.info "Processing team: #{team.name} (ID: #{team.id})"
  games = WhlGame.where("home_team_id = ? OR away_team_id = ?", team.hockeytech_id, team.hockeytech_id).order(:game_date)
      games.each_with_index do |game, idx|
        ks.each do |k|
          next if idx < k
          window = games[(idx-k)...idx]
          target_stats = {
            goals: [], ppp: [], sog: [], fowp: [], home_away: []
          }
          opponent_stats = {
            goals: [], ppp: [], sog: [], fowp: []
          }
          window.each do |g|
            is_home = g.home_team_id == team.hockeytech_id
            target_goals = is_home ? g.home_goal_count : g.away_goal_count
            opponent_goals = is_home ? g.away_goal_count : g.home_goal_count
            target_ppp = is_home ? g.home_power_play_percentage : g.away_power_play_percentage
            opponent_ppp = is_home ? g.away_power_play_percentage : g.home_power_play_percentage
            target_sog = is_home ? g.home_shots_on_goal_total : g.away_shots_on_goal_total
            opponent_sog = is_home ? g.away_shots_on_goal_total : g.home_shots_on_goal_total
            target_fowp = is_home ? g.home_faceoff_win_percentage : g.away_faceoff_win_percentage
            opponent_fowp = is_home ? g.away_faceoff_win_percentage : g.home_faceoff_win_percentage
            target_stats[:goals] << target_goals
            target_stats[:ppp] << target_ppp
            target_stats[:sog] << target_sog
            target_stats[:fowp] << target_fowp
            target_stats[:home_away] << (is_home ? 1 : 0)
            opponent_stats[:goals] << opponent_goals
            opponent_stats[:ppp] << opponent_ppp
            opponent_stats[:sog] << opponent_sog
            opponent_stats[:fowp] << opponent_fowp
          end
          avg = ->(arr) { arr.compact.size > 0 ? arr.compact.sum.to_f / arr.compact.size : nil }
          goals_diff = avg.call(target_stats[:goals]) - avg.call(opponent_stats[:goals])
          ppp_diff = avg.call(target_stats[:ppp]) - avg.call(opponent_stats[:ppp])
          sog_diff = avg.call(target_stats[:sog]) - avg.call(opponent_stats[:sog])
          fowp_diff = avg.call(target_stats[:fowp]) - avg.call(opponent_stats[:fowp])
          # Determine if the team won this game
          is_home = game.home_team_id == team.hockeytech_id
          team_goals = is_home ? game.home_goal_count : game.away_goal_count
          opp_goals = is_home ? game.away_goal_count : game.home_goal_count
          target_win = nil
          if !team_goals.nil? && !opp_goals.nil?
            target_win = team_goals > opp_goals ? 1 : 0
          end
          rolling_avg = WhlRollingAverage.find_or_initialize_by(
            game_id: game.game_id,
            whl_team_id: team.id,
            k_value: k
          )

          rolling_avg.update!(
            goals_for_avg: avg.call(target_stats[:goals]),
            goals_against_avg: avg.call(opponent_stats[:goals]),
            shots_for_avg: avg.call(target_stats[:sog]),
            shots_against_avg: avg.call(opponent_stats[:sog]),
            power_play_percentage_avg: avg.call(target_stats[:ppp]),
            power_play_percentage_against_avg: avg.call(opponent_stats[:ppp]),
            faceoff_win_percentage_avg: avg.call(target_stats[:fowp]),
            faceoff_win_percentage_against_avg: avg.call(opponent_stats[:fowp]),
            home_away: is_home ? 1 : 0,
            goals_diff: goals_diff,
            ppp_diff: ppp_diff,
            sog_diff: sog_diff,
            fowp_diff: fowp_diff,
            target_win: target_win
          )
        end
      end
    end
    logger.info "Rolling averages calculation complete."
  end
end
