class Api::V1::ChlController < ApplicationController

    def index
        @chl = WhlTeamStat.calc_rolling_average(params[:team_name], params[:window_size].to_i)

        @test = WhlTeamStat.calc_last_k_avg_stats(params[:team_name], params[:window_size].to_i)
        render json: { rolling_averages: @chl, last_k_stats: @test }
    end
end
