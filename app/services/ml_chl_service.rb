class MlChlService
  include HTTParty
  base_uri('http://localhost:2718')

  def self.calc_and_get_prediction(payload)
    # Get the results of prediction ML calculation
    predict_probs = self.post("/whl/calc_winner", body: payload.to_json, headers: { 'Content-Type' => 'application/json' })

    # Have home_team be first, before the away_team
    ordered = ActiveSupport::OrderedHash.new
    ordered['home_team_prob'] = predict_probs['home_team_prob']
    ordered['away_team_prob'] = predict_probs['away_team_prob']

    ordered
  end
end
