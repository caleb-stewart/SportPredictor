class AddHomeAndAwayTeamInfoToWhlGames < ActiveRecord::Migration[8.0]
  def change
    change_column :whl_teams, :hockeytech_id, :integer, using: 'hockeytech_id::integer'
    add_column :whl_games, :home_team, :string
    add_column :whl_games, :away_team, :string
    add_column :whl_games, :home_team_id, :integer
    add_column :whl_games, :away_team_id, :integer
    add_foreign_key :whl_games, :whl_teams, column: :home_team_id, primary_key: :hockeytech_id
    add_foreign_key :whl_games, :whl_teams, column: :away_team_id, primary_key: :hockeytech_id
    add_index :whl_games, :home_team_id
    add_index :whl_games, :away_team_id
  end
end
