class AddGameNumberScoreShotsToWhlGames < ActiveRecord::Migration[7.0]
  def change
    add_column :whl_games, :game_number, :integer
    add_column :whl_games, :scoring, :jsonb
    add_column :whl_games, :power_play, :jsonb
    add_column :whl_games, :fow, :jsonb
  end
end
