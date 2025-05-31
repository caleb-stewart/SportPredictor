class Api::V1::ChlController < ApplicationController

    def index
        @home_team_stats = WhlTeamStat.calc_rolling_average(params[:home_team], params[:window_size].to_i)
        @away_team_stats = WhlTeamStat.calc_rolling_average(params[:away_team], params[:window_size].to_i)

        @home_latest_stats = WhlTeamStat.calc_last_k_avg_stats(params[:home_team], params[:window_size].to_i, 1)
        @away_latest_stats = WhlTeamStat.calc_last_k_avg_stats(params[:away_team], params[:window_size].to_i, 0)
        prediction_payload = {
            past_stats: {
            home_team: @home_team_stats,
            away_team: @away_team_stats
            },
            predict_game: {
            home_team: @home_latest_stats,
            away_team: @away_latest_stats
            }
        }


        @win_probs = MlChlService.calc_and_get_prediction(prediction_payload)
        
        render json: { win_probs: @win_probs, prediction_features: {home_team: @home_latest_stats, away_team: @away_latest_stats} }
    end
end
