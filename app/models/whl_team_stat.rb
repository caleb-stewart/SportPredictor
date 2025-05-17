class WhlTeamStat < ApplicationRecord

    def self.calc_rolling_average(target_team = "Spokane Chiefs", window_size = 5)
        # access the db to get all the stats for the target team, either home or away
        target_team_stats = WhlTeamStat.where("home_name = ? OR away_name = ?", target_team, target_team).order(:game_id)

        # array to store the rolling averages
        averages = []

        # go through the stats and calculate the rolling averages
        target_team_stats.each_cons(window_size) do |window|
            # temp vars to store the avg stats for the target team and opponent
            target_goals = 0.0
            target_ppp = 0.0
            target_fowp = 0.0
            target_sog = 0.0

            opp_goals = 0.0
            opp_ppp = 0.0
            opp_fowp = 0.0
            opp_sog = 0.0

            # go through the window of games and calculate the stats
            window.each do |game|
                if game.home_name == target_team
                    target_goals += game.home_goals
                    target_ppp += game.home_ppp
                    target_fowp += game.home_fowp
                    target_sog += game.home_sog

                    opp_goals += game.away_goals
                    opp_ppp += game.away_ppp
                    opp_fowp += game.away_fowp
                    opp_sog += game.away_sog
                else
                    target_goals += game.away_goals
                    target_ppp += game.away_ppp
                    target_fowp += game.away_fowp
                    target_sog += game.away_sog

                    opp_goals += game.home_goals
                    opp_ppp += game.home_ppp
                    opp_fowp += game.home_fowp
                    opp_sog += game.home_sog
                end
            end
        
        # Get the last game in the window to determine if it was a home or away game
        last_game = window.last
        is_home = last_game.home_name == target_team
        home_away = is_home ? 1.0 : 0.0
        target_win = (
            (is_home && last_game.home_goals > last_game.away_goals) ||
            (!is_home && last_game.away_goals > last_game.home_goals)
        ) ? 1 : 0

        averages << {
            target_goals: target_goals / window_size,
            opponent_goals: opp_goals / window_size,
            target_ppp: target_ppp / window_size,
            opponent_ppp: opp_ppp / window_size,
            target_sog: target_sog / window_size,
            opponent_sog: opp_sog / window_size,
            target_fowp: target_fowp / window_size,
            opponent_fowp: opp_fowp / window_size,
            home_away: home_away,
            goals_diff: (target_goals - opp_goals) / window_size,
            ppp_diff: (target_ppp - opp_ppp) / window_size,
            sog_diff: (target_sog - opp_sog) / window_size,
            fowp_diff: (target_fowp - opp_fowp) / window_size,
            target_win: target_win
        }
        end

    # return the averages
    averages
  end

  def self.calc_last_k_avg_stats(target_team = "Spokane Chiefs", k = 5)
    # Fetch the last K games involving the target team
    target_team_stats = WhlTeamStat.where("home_name = ? OR away_name = ?", target_team, target_team).order(:game_id).last(k)

    # Initialize totals
    target_goals = 0.0
    target_ppp = 0.0
    target_fowp = 0.0
    target_sog = 0.0

    opp_goals = 0.0
    opp_ppp = 0.0
    opp_fowp = 0.0
    opp_sog = 0.0

    home_away = 0.0
    wins = 0

    target_team_stats.each do |game|
        is_home = game.home_name == target_team
        home_away += is_home ? 1.0 : 0.0

        if is_home
            target_goals += game.home_goals
            target_ppp   += game.home_ppp
            target_fowp  += game.home_fowp
            target_sog   += game.home_sog

            opp_goals += game.away_goals
            opp_ppp   += game.away_ppp
            opp_fowp  += game.away_fowp
            opp_sog   += game.away_sog

            wins += 1 if game.home_goals > game.away_goals
        else
            target_goals += game.away_goals
            target_ppp   += game.away_ppp
            target_fowp  += game.away_fowp
            target_sog   += game.away_sog

            opp_goals += game.home_goals
            opp_ppp   += game.home_ppp
            opp_fowp  += game.home_fowp
            opp_sog   += game.home_sog

        end
    end

    # Return averages
    {
        target_goals: target_goals / k,
        opponent_goals: opp_goals / k,
        target_ppp: target_ppp / k,
        opponent_ppp: opp_ppp / k,
        target_sog: target_sog / k,
        opponent_sog: opp_sog / k,
        target_fowp: target_fowp / k,
        opponent_fowp: opp_fowp / k,
        home_away: 0,
        goals_diff: (target_goals - opp_goals) / k,
        ppp_diff: (target_ppp - opp_ppp) / k,
        sog_diff: (target_sog - opp_sog) / k,
        fowp_diff: (target_fowp - opp_fowp) / k,
    }
    end

end
