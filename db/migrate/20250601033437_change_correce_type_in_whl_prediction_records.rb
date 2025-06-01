class ChangeCorreceTypeInWhlPredictionRecords < ActiveRecord::Migration[8.0]
  def change
    change_column :whl_prediction_records, :correct, :boolean
  end
end
