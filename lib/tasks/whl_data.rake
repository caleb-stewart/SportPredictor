# lib/tasks/whl_data.rake
namespace :whl_data do
  desc "Import historical WHL game data for all teams"
  task import: :environment do
    require "logger"
    logger = Logger.new(STDOUT)
    service = WhlApiService.new
    teams = WhlTeam.all
    delay_range = 0.1..0.5
    processed_games = Set.new

    # Team name mapping for canonical names
    TEAM_NAME_MAPPING = {
      "Kootenay ICE" => "Wenatchee Wild",
      "Winnipeg ICE" => "Wenatchee Wild",
      "Wenatchee Wild" => "Wenatchee Wild"
    }
    def canonical_team_name(name)
      TEAM_NAME_MAPPING[name] || name
    end

    batch_sizes = [ 1000, 2000 ]
    teams.each do |team|
      logger.info "Fetching games for team: #{team.name} (ID: #{team.hockeytech_id})"
      all_games = []
      batch_sizes.each do |batch|
        response = service.game_id_url(0, batch, team.hockeytech_id)
        data = response.parsed_response
        games = data.dig("Scoreboard", "Games") || []
        logger.info "Batch #{batch}: Found #{games.size} games for #{team.name}"
        all_games += games
      end
      # Remove duplicate games by ID
      all_games.uniq! { |g| g["ID"] || g["game_id"] }

      all_games.each_with_index do |game, idx|
        game_id = game["ID"] || game["game_id"]
        next if processed_games.include?(game_id) || WhlGame.exists?(game_id: game_id)
        logger.info "[#{team.name}] Processing game #{idx+1}/#{all_games.size}: #{game_id}"
        begin
          stats_response = service.get_game_stats_url(game_id)
          stats_data = stats_response.parsed_response
          clock = stats_data.dig("GC", "Clock") || {}

          # Use canonical team names
          home_team_name = canonical_team_name(clock.dig("home_team", "name") || game["HomeTeamName"])
          away_team_name = canonical_team_name(clock.dig("visiting_team", "name") || game["VisitorTeamName"])
          home_team = WhlTeam.find_by(name: home_team_name)
          away_team = WhlTeam.find_by(name: away_team_name)

          WhlGame.create!(
            game_id: game_id,
            season_id: clock["season_id"] || game["SeasonID"],
            season_name: clock["season_name"] || game["SeasonName"],
            game_date_iso_8601: clock["game_date_iso_8601"] || game["GameDateISO8601"],
            venue: clock["venue"] || game["venue_name"],
            status: clock["status"] || game["GameStatusString"],
            home_team_id: home_team&.id,
            away_team_id: away_team&.id,
            home_goal_count: clock["home_goal_count"]&.to_i || game["HomeGoals"]&.to_i,
            away_goal_count: clock["visiting_goal_count"]&.to_i || game["VisitorGoals"]&.to_i,
            scoring_breakdown: clock["scoring"],
            shots_on_goal: clock["shots_on_goal"],
            home_power_play_attempts: clock.dig("power_play", "total", "home")&.to_i,
            away_power_play_attempts: clock.dig("power_play", "total", "visiting")&.to_i,
            home_power_play_goals: clock.dig("power_play", "goals", "home")&.to_i,
            away_power_play_goals: clock.dig("power_play", "goals", "visiting")&.to_i,
            home_faceoffs_won: clock.dig("fow", "home")&.to_i,
            away_faceoffs_won: clock.dig("fow", "visiting")&.to_i,
            period: clock["period"] || game["Period"],
            game_clock: clock["game_clock"] || game["GameClock"],
            game_number: clock["game_number"]&.to_i,
            scoring: clock["scoring"],
            power_play: clock["power_play"],
            fow: clock["fow"]
          )
          processed_games << game_id
        rescue => e
          logger.error "Error processing game #{game_id}: #{e.message}"
        end
        sleep(rand(delay_range)) # Rate limiting
      end
    end
    logger.info "WHL historical data import complete."
  end
end
