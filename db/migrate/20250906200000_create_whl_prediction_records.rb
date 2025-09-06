class CreateWhlPredictionRecords < ActiveRecord::Migration[7.1]
  def change
    create_table :whl_prediction_records do |t|
      t.string :game_id, null: false
      t.integer :k_value, null: false
      t.references :home_team, null: false, foreign_key: { to_table: :whl_teams }
      t.references :away_team, null: false, foreign_key: { to_table: :whl_teams }
      t.references :predicted_winner, foreign_key: { to_table: :whl_teams }
      t.decimal :home_team_probability, precision: 5, scale: 4
      t.decimal :away_team_probability, precision: 5, scale: 4
      t.references :actual_winner, foreign_key: { to_table: :whl_teams }
      t.boolean :correct, default: nil
      t.datetime :prediction_date
      t.timestamps
    end
    add_index :whl_prediction_records, [ :game_id, :k_value ], unique: true
    add_index :whl_prediction_records, :prediction_date
  end
end
