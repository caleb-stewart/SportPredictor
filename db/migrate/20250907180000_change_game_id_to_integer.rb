class ChangeGameIdToInteger < ActiveRecord::Migration[8.0]
  def up
    change_column :whl_games, :game_id, :integer, using: 'game_id::integer'
    change_column :whl_prediction_records, :game_id, :integer, using: 'game_id::integer'
    change_column :whl_rolling_averages, :game_id, :integer, using: 'game_id::integer'
  end

  def down
    change_column :whl_games, :game_id, :string
    change_column :whl_prediction_records, :game_id, :string
    change_column :whl_rolling_averages, :game_id, :string
  end
end
