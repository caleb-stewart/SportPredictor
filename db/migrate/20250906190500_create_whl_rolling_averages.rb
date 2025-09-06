class CreateWhlRollingAverages < ActiveRecord::Migration[7.1]
  def change
    create_table :whl_rolling_averages do |t|
      t.string :game_id, null: false
      t.references :whl_team, null: false, foreign_key: true
      t.integer :k_value, null: false
      t.decimal :goals_for_avg, precision: 8, scale: 2
      t.decimal :goals_against_avg, precision: 8, scale: 2
      t.decimal :shots_for_avg, precision: 8, scale: 2
      t.decimal :shots_against_avg, precision: 8, scale: 2
      t.decimal :power_play_percentage_avg, precision: 8, scale: 2
      t.decimal :penalty_kill_percentage_avg, precision: 8, scale: 2
      t.timestamps
    end
    add_index :whl_rolling_averages, [ :game_id, :k_value ], unique: true
    add_index :whl_rolling_averages, [ :whl_team_id, :k_value ]
  end
end
