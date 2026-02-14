module Api
  module V1
    class PredictionHistoryController < ApplicationController
      def history
        predictions = WhlPredictionRecord
          .includes(:home_team, :away_team)
          .order(created_at: :desc)
          .limit(300)

        render json: format_predictions(predictions)
      end

      private

      def format_predictions(predictions)
        grouped = predictions.group_by(&:game_id)

        grouped.map do |game_id, records|
          first = records.first

          {
            game_id: game_id,
            home_team: first.home_team&.name,
            away_team: first.away_team&.name,
            model_version: first.model_version,
            model_family: first.model_family,
            k_values: records.sort_by(&:k_value).map do |record|
              {
                k_value: record.k_value.to_s,
                home_prob: record.home_team_probability,
                away_prob: record.away_team_probability,
                correct: record.correct
              }
            end
          }
        end
      end
    end
  end
end
