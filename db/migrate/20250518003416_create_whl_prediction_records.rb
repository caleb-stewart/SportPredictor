class CreateWhlPredictionRecords < ActiveRecord::Migration[8.0]
  def change
    create_table :whl_prediction_records do |t|
      t.integer :game_id
      t.integer :k_value
      t.float :home_prob
      t.float :away_prob
      t.integer :correct
      t.string :home_team
      t.string :away_team

      t.timestamps
    end
  end
end
