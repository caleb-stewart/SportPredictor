class MlChlService
  include HTTParty
  base_uri('http://localhost:2718')

  def self.calc_and_get_prediction(payload)

    self.post("/whl/calc_winner", body: payload.to_json, headers: { 'Content-Type' => 'application/json' })
  end
end
