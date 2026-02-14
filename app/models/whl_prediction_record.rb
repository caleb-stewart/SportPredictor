class WhlPredictionRecord < ApplicationRecord
  belongs_to :home_team, class_name: "WhlTeam"
  belongs_to :away_team, class_name: "WhlTeam"
  belongs_to :predicted_winner, class_name: "WhlTeam", optional: true
  belongs_to :actual_winner, class_name: "WhlTeam", optional: true

  scope :scored, -> { where.not(correct: nil) }

  # Backward compatibility for stale code paths.
  def home_prob
    home_team_probability
  end

  def away_prob
    away_team_probability
  end
end
