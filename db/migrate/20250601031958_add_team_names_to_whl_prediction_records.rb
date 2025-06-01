class AddTeamNamesToWhlPredictionRecords < ActiveRecord::Migration[8.0]
  def change
    add_column :whl_prediction_records, :home_team, :string
    add_column :whl_prediction_records, :away_team, :string
  end
end
