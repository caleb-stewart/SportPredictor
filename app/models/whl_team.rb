class WhlTeam < ApplicationRecord
  has_many :home_predictions, class_name: "WhlPredictionRecord", foreign_key: :home_team_id
  has_many :away_predictions, class_name: "WhlPredictionRecord", foreign_key: :away_team_id
  has_many :predicted_wins, class_name: "WhlPredictionRecord", foreign_key: :predicted_winner_id
  has_many :actual_wins, class_name: "WhlPredictionRecord", foreign_key: :actual_winner_id
end
