class ModifyWhlRollingAverages < ActiveRecord::Migration[8.0]
  def change
    remove_index :whl_rolling_averages, [ :game_id, :k_value ], name: "index_whl_rolling_averages_on_game_id_and_k_value"
    add_index :whl_rolling_averages, [ :game_id, :k_value, :whl_team_id ], unique: true, name: "index_whl_rolling_averages_on_game_id_k_value_whl_team_id"
    drop_table :whl_team_stats
  end
end
