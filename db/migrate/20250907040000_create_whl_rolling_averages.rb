class CreateWhlRollingAverages < ActiveRecord::Migration[8.0]
  def change
    create_table :whl_rolling_averages do |t|
      t.string :game_id, null: false
      t.references :whl_team, null: false, foreign_key: true
      t.integer :k_value, null: false
      t.decimal :goals_for_avg, precision: 8, scale: 2
      t.decimal :goals_against_avg, precision: 8, scale: 2
      t.decimal :shots_for_avg, precision: 8, scale: 2
      t.decimal :shots_against_avg, precision: 8, scale: 2
      t.decimal :power_play_percentage_avg, precision: 8, scale: 4
      t.decimal :power_play_percentage_against_avg, precision: 8, scale: 4
      t.decimal :faceoff_win_percentage_avg, precision: 8, scale: 4
      t.decimal :faceoff_win_percentage_against_avg, precision: 8, scale: 4
      t.integer :home_away
      t.decimal :goals_diff, precision: 8, scale: 2
      t.decimal :ppp_diff, precision: 8, scale: 4
      t.decimal :sog_diff, precision: 8, scale: 2
      t.decimal :fowp_diff, precision: 8, scale: 4
      t.timestamps
    end
    add_index :whl_rolling_averages, [ :game_id, :k_value, :whl_team_id ], unique: true, name: "index_whl_rolling_averages_on_game_id_k_value_whl_team_id"
  end
end
