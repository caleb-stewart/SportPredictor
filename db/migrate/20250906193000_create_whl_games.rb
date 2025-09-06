class CreateWhlGames < ActiveRecord::Migration[7.1]
  def change
    create_table :whl_games do |t|
      t.string :game_id, null: false
      t.string :season_id
      t.string :season_name
      t.datetime :game_date_iso_8601
      t.string :venue
      t.string :status
      t.references :home_team, null: false, foreign_key: { to_table: :whl_teams }
      t.references :away_team, null: false, foreign_key: { to_table: :whl_teams }
      t.integer :home_goal_count
      t.integer :away_goal_count
      t.jsonb :scoring_breakdown # { home: {1: x, 2: y, 3: z}, away: {...} }
      t.jsonb :shots_on_goal # { home: {1: x, 2: y, 3: z}, away: {...} }
      t.integer :home_power_play_attempts
      t.integer :away_power_play_attempts
      t.integer :home_power_play_goals
      t.integer :away_power_play_goals
      t.integer :home_faceoffs_won
      t.integer :away_faceoffs_won
      t.string :period
      t.string :game_clock
      t.timestamps
    end
    add_index :whl_games, :game_id, unique: true
  end
end
