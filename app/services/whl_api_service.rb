class WhlApiService
  # How we make HTTP requests to external APIs
  include HTTParty
  # class method provided by HTTParty to set the base URI for all requests in this class
  base_uri("https://lscluster.hockeytech.com/feed/")

  def initialize
    # Secret key, that is actually public from the WHL site
    @key = ENV["HOCKEYTECH_API"]

    # Parameters always used in the URL
    @default_params = {
      client_code: "whl",
      fmt: "json",
      lang_code: "en",
      key: @key
    }
  end

  def game_id_url(num_of_days_ahead = 0, num_of_past_games = 0, current_team_id = "")
    # This is all of the parameters that are used in the URL
    query = @default_params.merge({
      feed: "modulekit",
      view: "scorebar",
      numberofdaysahead: num_of_days_ahead,
      numberofdaysback: num_of_past_games,
      season_id: "",
      team_id: current_team_id
    })

    # HTTParty appends '' to base_uri
    # and then appends the query string to that
    # so we need to pass an empty string to the get method
    self.class.get("", query: query)
  end

  def get_game_stats_url(game_id)
    # Append the parameters to the base URI
    query = @default_params.merge({
      feed: "gc",
      game_id: game_id,
      tab: "clock"
    })

    # HTTParty appends '' to base_uri
    # and then appends the query string to that
    # so we need to pass an empty string to the get method
    self.class.get("", query: query)
  end
end
