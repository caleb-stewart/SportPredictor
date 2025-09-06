class CreateWhlTeamStats < ActiveRecord::Migration[7.1]
  def change
    create_table :whl_team_stats do |t|
      t.string :game_id, null: false
      t.references :home_team, null: false, foreign_key: { to_table: :whl_teams }
      t.references :away_team, null: false, foreign_key: { to_table: :whl_teams }
      t.boolean :completed, default: false
      t.string :season_id
      t.string :season_name
      t.datetime :game_date_iso_8601
      t.integer :home_goals
      t.integer :away_goals
      t.integer :home_shots
      t.integer :away_shots
      t.decimal :home_power_play_percentage, precision: 8, scale: 2
      t.decimal :away_power_play_percentage, precision: 8, scale: 2
      t.decimal :home_penalty_kill_percentage, precision: 8, scale: 2
      t.decimal :away_penalty_kill_percentage, precision: 8, scale: 2
      t.timestamps
    end
    add_index :whl_team_stats, :game_id, unique: true
  end
end
