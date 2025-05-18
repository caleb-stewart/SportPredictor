module Api
  module V1
    class PredictionHistoryController < ApplicationController

      def history
        
        predictions = WhlPredictionRecord.all.order(created_at: :desc).limit(100)
        puts predictions.to_json

        # game_ids = predictions.pluck(:game_id).uniq

        # home_team_name = WhlTeamStat.where(game_id: predictions[:game_id])
        # puts home_team_name

        # 
        render json: predictions
      end
    end
  end
end