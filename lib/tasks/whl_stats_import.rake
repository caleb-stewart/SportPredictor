# lib/tasks/whl_import.rake
require "csv"

namespace :import do
  desc "Import WHL team stats from CSV file"
  task whl_stats: :environment do
    csv_path = Rails.root.join("db", "data", "All_teams_WHL_stats.csv")

    unless File.exist?(csv_path)
      puts "CSV file not found at #{csv_path}"
      next
    end

    puts "Importing WHL team stats from #{csv_path}..."
    CSV.foreach(csv_path, headers: true, header_converters: :symbol) do |row|
        if !WhlTeamStat.exists?(game_id: row[:game_id])
          puts "Updating with game: ", row[:game_id]
          WhlTeamStat.create!(
            game_id: row[:game_id],
            home_name: row[:home_name],
            away_name: row[:away_name],
            home_goals: row[:home_goals],
            away_goals: row[:away_goals],
            home_ppp: row[:home_pp],
            away_ppp: row[:away_pp],
            home_sog: row[:home_sog],
            away_sog: row[:away_sog],
            home_fowp: row[:home_fow],
            away_fowp: row[:away_fow]
          )
        end
    end

    puts "WHL team stats successfully imported!"
  end
end
