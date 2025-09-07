class AddPredictorFeaturesToWhlRollingAverages < ActiveRecord::Migration[8.0]
  def change
    add_column :whl_rolling_averages, :goals_diff, :decimal, precision: 8, scale: 2
    add_column :whl_rolling_averages, :ppp_diff, :decimal, precision: 8, scale: 4
    add_column :whl_rolling_averages, :sog_diff, :decimal, precision: 8, scale: 2
    add_column :whl_rolling_averages, :fowp_diff, :decimal, precision: 8, scale: 4
    add_column :whl_rolling_averages, :target_fowp, :decimal, precision: 8, scale: 4
    add_column :whl_rolling_averages, :opponent_fowp, :decimal, precision: 8, scale: 4
    add_column :whl_rolling_averages, :home_away, :integer
    remove_column :whl_rolling_averages, :penalty_kill_percentage_avg
  end
end
