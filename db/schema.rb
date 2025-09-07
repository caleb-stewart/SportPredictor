# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.0].define(version: 2025_09_07_203000) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "whl_games", force: :cascade do |t|
    t.integer "game_id", null: false
    t.string "season_id"
    t.string "season_name"
    t.date "game_date"
    t.string "venue"
    t.string "status"
    t.integer "home_goal_count"
    t.integer "away_goal_count"
    t.jsonb "scoring_breakdown"
    t.jsonb "shots_on_goal"
    t.string "period"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer "game_number"
    t.jsonb "power_play"
    t.jsonb "fow"
    t.decimal "home_power_play_percentage", precision: 8, scale: 4
    t.decimal "away_power_play_percentage", precision: 8, scale: 4
    t.decimal "home_faceoff_win_percentage", precision: 8, scale: 4
    t.decimal "away_faceoff_win_percentage", precision: 8, scale: 4
    t.integer "home_shots_on_goal_total"
    t.integer "away_shots_on_goal_total"
    t.string "home_team"
    t.string "away_team"
    t.integer "home_team_id"
    t.integer "away_team_id"
    t.index ["away_team_id"], name: "index_whl_games_on_away_team_id"
    t.index ["game_id"], name: "index_whl_games_on_game_id", unique: true
    t.index ["home_team_id"], name: "index_whl_games_on_home_team_id"
  end

  create_table "whl_prediction_records", force: :cascade do |t|
    t.integer "game_id", null: false
    t.integer "k_value", null: false
    t.bigint "home_team_id", null: false
    t.bigint "away_team_id", null: false
    t.bigint "predicted_winner_id"
    t.decimal "home_team_probability", precision: 5, scale: 4
    t.decimal "away_team_probability", precision: 5, scale: 4
    t.bigint "actual_winner_id"
    t.boolean "correct"
    t.datetime "prediction_date"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["actual_winner_id"], name: "index_whl_prediction_records_on_actual_winner_id"
    t.index ["away_team_id"], name: "index_whl_prediction_records_on_away_team_id"
    t.index ["game_id", "k_value"], name: "index_whl_prediction_records_on_game_id_and_k_value", unique: true
    t.index ["home_team_id"], name: "index_whl_prediction_records_on_home_team_id"
    t.index ["predicted_winner_id"], name: "index_whl_prediction_records_on_predicted_winner_id"
    t.index ["prediction_date"], name: "index_whl_prediction_records_on_prediction_date"
  end

  create_table "whl_rolling_averages", force: :cascade do |t|
    t.integer "game_id", null: false
    t.bigint "whl_team_id", null: false
    t.integer "k_value", null: false
    t.decimal "goals_for_avg", precision: 8, scale: 2
    t.decimal "goals_against_avg", precision: 8, scale: 2
    t.decimal "shots_for_avg", precision: 8, scale: 2
    t.decimal "shots_against_avg", precision: 8, scale: 2
    t.decimal "power_play_percentage_avg", precision: 8, scale: 4
    t.decimal "power_play_percentage_against_avg", precision: 8, scale: 4
    t.decimal "faceoff_win_percentage_avg", precision: 8, scale: 4
    t.decimal "faceoff_win_percentage_against_avg", precision: 8, scale: 4
    t.integer "home_away"
    t.decimal "goals_diff", precision: 8, scale: 2
    t.decimal "ppp_diff", precision: 8, scale: 4
    t.decimal "sog_diff", precision: 8, scale: 2
    t.decimal "fowp_diff", precision: 8, scale: 4
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.integer "target_win"
    t.index ["game_id", "k_value", "whl_team_id"], name: "index_whl_rolling_averages_on_game_id_k_value_whl_team_id", unique: true
    t.index ["whl_team_id"], name: "index_whl_rolling_averages_on_whl_team_id"
  end

  create_table "whl_teams", force: :cascade do |t|
    t.string "name", null: false
    t.integer "hockeytech_id", null: false
    t.string "city"
    t.string "team_name"
    t.string "conference"
    t.string "division"
    t.boolean "active", default: true
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["hockeytech_id"], name: "index_whl_teams_on_hockeytech_id", unique: true
    t.index ["name"], name: "index_whl_teams_on_name", unique: true
  end

  add_foreign_key "whl_games", "whl_teams", column: "away_team_id", primary_key: "hockeytech_id"
  add_foreign_key "whl_games", "whl_teams", column: "home_team_id", primary_key: "hockeytech_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "actual_winner_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "away_team_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "home_team_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "predicted_winner_id"
  add_foreign_key "whl_rolling_averages", "whl_teams"
end
