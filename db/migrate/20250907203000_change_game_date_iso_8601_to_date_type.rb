class ChangeGameDateIso8601ToDateType < ActiveRecord::Migration[8.0]
  def up
    rename_column :whl_games, :game_date_iso_8601, :game_date
    change_column :whl_games, :game_date, :date
  end

  def down
    change_column :whl_games, :game_date, :datetime
    rename_column :whl_games, :game_date, :game_date_iso_8601
  end
end
