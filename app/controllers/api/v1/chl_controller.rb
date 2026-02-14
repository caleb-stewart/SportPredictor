class Api::V1::ChlController < ApplicationController
  def index
    home_team = WhlTeam.find_by(name: params[:home_team])
    away_team = WhlTeam.find_by(name: params[:away_team])

    unless home_team && away_team
      return render json: { error: "Unknown home_team or away_team" }, status: :unprocessable_entity
    end

    matchup_date = params[:game_date].present? ? Date.parse(params[:game_date]) : Date.current + 1
    game = WhlGame.new(
      game_id: nil,
      game_date: matchup_date,
      home_team_id: home_team.hockeytech_id,
      away_team_id: away_team.hockeytech_id,
      home_team: home_team.name,
      away_team: away_team.name
    )

    payload = WhlPredictionFeatureService.build_payload(game: game)
    unless payload
      return render json: { error: "Insufficient rolling history for one or more k values" }, status: :unprocessable_entity
    end

    prediction = MlChlService.calc_and_get_prediction(payload)

    render json: {
      win_probs: {
        home_team_prob: prediction["home_team_prob"],
        away_team_prob: prediction["away_team_prob"]
      },
      prediction_features: payload[:features_by_k],
      model_version: prediction["model_version"],
      model_family: prediction["model_family"],
      k_components: prediction["k_components"]
    }
  rescue ArgumentError
    render json: { error: "Invalid game_date format. Use YYYY-MM-DD" }, status: :unprocessable_entity
  end
end
