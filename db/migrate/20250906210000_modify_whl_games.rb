class ModifyWhlGames < ActiveRecord::Migration[7.0]
  def change
    add_column :whl_games, :game_number, :integer
    add_column :whl_games, :power_play, :jsonb
    add_column :whl_games, :fow, :jsonb
    add_column :whl_games, :home_power_play_percentage, :decimal, precision: 8, scale: 4
    add_column :whl_games, :away_power_play_percentage, :decimal, precision: 8, scale: 4
    add_column :whl_games, :home_faceoff_win_percentage, :decimal, precision: 8, scale: 4
    add_column :whl_games, :away_faceoff_win_percentage, :decimal, precision: 8, scale: 4
    add_column :whl_games, :home_shots_on_goal_total, :integer
    add_column :whl_games, :away_shots_on_goal_total, :integer

    remove_column :whl_games, :home_team_id
    remove_column :whl_games, :away_team_id
    remove_column :whl_games, :home_power_play_attempts
    remove_column :whl_games, :away_power_play_attempts
    remove_column :whl_games, :home_power_play_goals
    remove_column :whl_games, :away_power_play_goals
    remove_column :whl_games, :game_clock
    remove_column :whl_games, :home_faceoffs_won
    remove_column :whl_games, :away_faceoffs_won
  end
end
