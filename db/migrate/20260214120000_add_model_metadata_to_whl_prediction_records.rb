class AddModelMetadataToWhlPredictionRecords < ActiveRecord::Migration[8.0]
  def change
    add_column :whl_prediction_records, :model_version, :string
    add_column :whl_prediction_records, :model_family, :string
    add_column :whl_prediction_records, :raw_model_outputs, :jsonb

    add_index :whl_prediction_records, :model_version
  end
end
