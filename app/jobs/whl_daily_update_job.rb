class WhlDailyUpdateJob < ApplicationJob
  queue_as :default

  # Fetches yesterday's finished games and updates whl_games + prediction correctness.
  def perform(*_args)
    logger = Logger.new(STDOUT)
    whl_api_service = WhlApiService.new

    update_games = whl_api_service.game_id_url(0, 1)
    update_games = update_games.dig("SiteKit", "Scorebar") || []

    update_games.each do |game_data|
      game_id = game_data["ID"]&.to_i
      next unless game_id

      begin
        stats_response = whl_api_service.get_game_stats_url(game_id)
        stats_data = stats_response.parsed_response
        clock = stats_data.dig("GC", "Clock") || {}

        whl_game = upsert_game_from_clock(game_id: game_id, clock: clock, fallback_game: game_data)
        next unless whl_game

        update_prediction_records(whl_game: whl_game, logger: logger)
      rescue => e
        logger.error "Daily update failed for game #{game_id}: #{e.message}"
      end
    end

    nil
  end

  private

  def upsert_game_from_clock(game_id:, clock:, fallback_game:)
    home_team_id = clock.dig("home_team", "team_id")&.to_i || fallback_game["HomeID"]&.to_i
    away_team_id = clock.dig("visiting_team", "team_id")&.to_i || fallback_game["VisitorID"]&.to_i

    home_pp_attempts = clock.dig("power_play", "total", "home")&.to_f
    away_pp_attempts = clock.dig("power_play", "total", "visiting")&.to_f
    home_pp_goals = clock.dig("power_play", "goals", "home")&.to_f || 0.0
    away_pp_goals = clock.dig("power_play", "goals", "visiting")&.to_f || 0.0

    home_power_play_percentage = home_pp_attempts.to_f.positive? ? (home_pp_goals / home_pp_attempts) : 0.0
    away_power_play_percentage = away_pp_attempts.to_f.positive? ? (away_pp_goals / away_pp_attempts) : 0.0

    home_fow = clock.dig("fow", "home")&.to_f || 0.0
    away_fow = clock.dig("fow", "visiting")&.to_f || 0.0
    total_fow = home_fow + away_fow

    home_faceoff_win_percentage = total_fow.positive? ? (home_fow / total_fow) : 0.5
    away_faceoff_win_percentage = total_fow.positive? ? (away_fow / total_fow) : 0.5

    home_sog = clock.dig("shots_on_goal", "home")&.values&.map(&:to_i)&.sum || 0
    away_sog = clock.dig("shots_on_goal", "visiting")&.values&.map(&:to_i)&.sum || 0

    game = WhlGame.find_or_initialize_by(game_id: game_id)
    game.assign_attributes(
      season_id: clock["season_id"] || fallback_game["SeasonID"],
      season_name: clock["season_name"] || fallback_game["SeasonName"],
      game_date: safe_iso8601_to_date(clock["game_date_iso_8601"] || fallback_game["GameDateISO8601"]),
      venue: clock["venue"] || fallback_game["venue_name"],
      status: clock["progress"] || fallback_game["GameStatusString"],
      home_team_id: home_team_id,
      away_team_id: away_team_id,
      home_team: clock.dig("home_team", "name") || fallback_game["HomeLongName"],
      away_team: clock.dig("visiting_team", "name") || fallback_game["VisitorLongName"],
      home_goal_count: clock["home_goal_count"]&.to_i,
      away_goal_count: clock["visiting_goal_count"]&.to_i,
      game_number: clock["game_number"]&.to_i,
      period: clock["period"],
      scoring_breakdown: clock["scoring"],
      shots_on_goal: clock["shots_on_goal"],
      power_play: clock["power_play"],
      fow: clock["fow"],
      home_power_play_percentage: home_power_play_percentage,
      away_power_play_percentage: away_power_play_percentage,
      home_faceoff_win_percentage: home_faceoff_win_percentage,
      away_faceoff_win_percentage: away_faceoff_win_percentage,
      home_shots_on_goal_total: home_sog,
      away_shots_on_goal_total: away_sog
    )

    game.save!
    game
  end

  def update_prediction_records(whl_game:, logger:)
    home_team = WhlTeam.find_by(hockeytech_id: whl_game.home_team_id)
    away_team = WhlTeam.find_by(hockeytech_id: whl_game.away_team_id)
    return unless home_team && away_team

    actual_winner_id = if whl_game.home_goal_count.to_i > whl_game.away_goal_count.to_i
      home_team.id
    elsif whl_game.home_goal_count.to_i < whl_game.away_goal_count.to_i
      away_team.id
    end

    WhlPredictionRecord.where(game_id: whl_game.game_id).find_each do |record|
      predicted_winner_id = record.home_team_probability.to_f >= record.away_team_probability.to_f ?
        record.home_team_id : record.away_team_id

      record.update!(
        actual_winner_id: actual_winner_id,
        predicted_winner_id: predicted_winner_id,
        correct: actual_winner_id.present? ? (predicted_winner_id == actual_winner_id) : nil
      )

      logger.info "Updated prediction record #{record.id} for game #{whl_game.game_id}"
    end
  end

  def safe_iso8601_to_date(value)
    return nil if value.blank?

    Date.iso8601(value)
  rescue ArgumentError
    nil
  end
end
