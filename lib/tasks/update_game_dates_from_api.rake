# lib/tasks/update_game_dates_from_api.rake
namespace :whl_data do
  desc "Update game_date_iso_8601 in whl_games from HockeyTech API (local date only)"
  task update_game_dates: :environment do
    require "logger"
    logger = Logger.new(STDOUT)
    service = WhlApiService.new
    teams = WhlTeam.all
    batch_sizes = [ 1000, 2000 ]
    updated = 0

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

      all_games.each do |game|
        game_id = game["ID"] || game["game_id"]
        game_date_str = game["GameDateISO8601"]
        next unless game_id && game_date_str
        # Extract only the date part from the ISO string
        local_date = Date.iso8601(game_date_str) rescue nil
        next unless local_date
        whl_game = WhlGame.find_by(game_id: game_id)
        if whl_game && whl_game.game_date != local_date
          whl_game.update!(game_date: local_date)
          updated += 1
        end
      end
    end
  logger.info "Updated game_date for #{updated} games."
  end
end
