# lib/tasks/whl_data.rake
namespace :whl_data do
  desc "Import historical WHL game data for all teams"
  task import: :environment do
    require "logger"
    logger = Logger.new(STDOUT)
    service = WhlApiService.new
    teams = WhlTeam.all
    delay_range = 0.1..0.25
    processed_games = Set.new

    # Team name mapping for canonical names
    TEAM_NAME_MAPPING = {
      "Kootenay ICE" => "Wenatchee Wild",
      "Winnipeg ICE" => "Wenatchee Wild",
      "Wenatchee Wild" => "Wenatchee Wild"
    }

    batch_sizes = [ 1000, 2000 ]
    teams.each do |team|
      logger.info "Fetching games for team: #{team.name} (ID: #{team.hockeytech_id})"
      all_games = []
      batch_sizes.each do |batch|
        response = service.game_id_url(0, batch, team.hockeytech_id)
        data = response.parsed_response
        games = data.dig("SiteKit", "Scorebar") || []
        logger.info "Batch #{batch}: Found #{games.size} games for #{team.name}"
        all_games += games
      end
      # Remove duplicate games by ID
      all_games.uniq! { |g| g["ID"] || g["game_id"] }

      all_games.each_with_index do |game, idx|
        home_team_id = game["HomeID"]
        away_team_id = game["VisitorID"]
        home_team = WhlTeam.find_by(hockeytech_id: home_team_id)
        away_team = WhlTeam.find_by(hockeytech_id: away_team_id)
        game_id = game["ID"] || game["game_id"]
        next if processed_games.include?(game_id) || WhlGame.exists?(game_id: game_id)
        logger.info "[#{team.name}] Processing game #{idx+1}/#{all_games.size}: #{game_id}"
        begin
          stats_response = service.get_game_stats_url(game_id)
          stats_data = stats_response.parsed_response
          clock = stats_data.dig("GC", "Clock") || {}

          # Calculate percentages and totals
          home_pp_attempts = clock.dig("power_play", "total", "home")&.to_f
          away_pp_attempts = clock.dig("power_play", "total", "visiting")&.to_f
          home_pp_goals = clock.dig("power_play", "goals", "home")&.to_f
          away_pp_goals = clock.dig("power_play", "goals", "visiting")&.to_f
          home_power_play_percentage = (home_pp_attempts && home_pp_attempts > 0) ? (home_pp_goals / home_pp_attempts) : 0.0
          away_power_play_percentage = (away_pp_attempts && away_pp_attempts > 0) ? (away_pp_goals / away_pp_attempts) : 0.0

          home_fow = clock.dig("fow", "home")&.to_f
          away_fow = clock.dig("fow", "visiting")&.to_f
          fow_total = (home_fow || 0) + (away_fow || 0)
          home_faceoff_win_percentage = (fow_total > 0) ? (home_fow / fow_total) : 0.5
          away_faceoff_win_percentage = (fow_total > 0) ? (away_fow / fow_total) : 0.5

          home_sog = clock.dig("shots_on_goal", "home")&.values&.map(&:to_i)&.sum || 0
          away_sog = clock.dig("shots_on_goal", "visiting")&.values&.map(&:to_i)&.sum || 0

          # Assign game_date as a Ruby Date object from GameDateISO8601 (like update_game_dates_from_api.rake)
          iso_date = clock["GameDateISO8601"] || game["GameDateISO8601"]
          game_date = iso_date ? (Date.iso8601(iso_date) rescue nil) : nil
          WhlGame.create!(
            game_id: game_id,
            season_id: clock["season_id"] || game["SeasonID"],
            season_name: clock["season_name"] || game["SeasonName"],
            game_date: game_date,
            venue: clock["venue"] || game["venue_name"],
            status: clock["status"] || game["GameStatusString"],
            home_team: home_team&.name || (game["HomeLongName"] || game["HomeTeamName"]),
            away_team: away_team&.name || (game["VisitorLongName"] || game["VisitorTeamName"]),
            home_team_id: home_team&.hockeytech_id,
            away_team_id: away_team&.hockeytech_id,
            home_goal_count: clock["home_goal_count"]&.to_i || game["HomeGoals"]&.to_i,
            away_goal_count: clock["visiting_goal_count"]&.to_i || game["VisitorGoals"]&.to_i,
            scoring_breakdown: clock["scoring"],
            shots_on_goal: clock["shots_on_goal"],
            period: clock["period"] || game["Period"],
            game_number: clock["game_number"]&.to_i || game["game_number"]&.to_i,
            power_play: clock["power_play"],
            fow: clock["fow"],
            home_power_play_percentage: home_power_play_percentage,
            away_power_play_percentage: away_power_play_percentage,
            home_faceoff_win_percentage: home_faceoff_win_percentage,
            away_faceoff_win_percentage: away_faceoff_win_percentage,
            home_shots_on_goal_total: home_sog,
            away_shots_on_goal_total: away_sog
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
