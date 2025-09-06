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

ActiveRecord::Schema[8.0].define(version: 2025_09_06_201000) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "whl_games", force: :cascade do |t|
    t.string "game_id", null: false
    t.string "season_id"
    t.string "season_name"
    t.datetime "game_date_iso_8601"
    t.string "venue"
    t.string "status"
    t.bigint "home_team_id", null: false
    t.bigint "away_team_id", null: false
    t.integer "home_goal_count"
    t.integer "away_goal_count"
    t.jsonb "scoring_breakdown"
    t.jsonb "shots_on_goal"
    t.integer "home_power_play_attempts"
    t.integer "away_power_play_attempts"
    t.integer "home_power_play_goals"
    t.integer "away_power_play_goals"
    t.integer "home_faceoffs_won"
    t.integer "away_faceoffs_won"
    t.string "period"
    t.string "game_clock"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["away_team_id"], name: "index_whl_games_on_away_team_id"
    t.index ["game_id"], name: "index_whl_games_on_game_id", unique: true
    t.index ["home_team_id"], name: "index_whl_games_on_home_team_id"
  end

  create_table "whl_prediction_records", force: :cascade do |t|
    t.string "game_id", null: false
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
    t.string "game_id", null: false
    t.bigint "whl_team_id", null: false
    t.integer "k_value", null: false
    t.decimal "goals_for_avg", precision: 8, scale: 2
    t.decimal "goals_against_avg", precision: 8, scale: 2
    t.decimal "shots_for_avg", precision: 8, scale: 2
    t.decimal "shots_against_avg", precision: 8, scale: 2
    t.decimal "power_play_percentage_avg", precision: 8, scale: 2
    t.decimal "penalty_kill_percentage_avg", precision: 8, scale: 2
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game_id", "k_value"], name: "index_whl_rolling_averages_on_game_id_and_k_value", unique: true
    t.index ["whl_team_id", "k_value"], name: "index_whl_rolling_averages_on_whl_team_id_and_k_value"
    t.index ["whl_team_id"], name: "index_whl_rolling_averages_on_whl_team_id"
  end

  create_table "whl_team_stats", force: :cascade do |t|
    t.string "game_id", null: false
    t.bigint "home_team_id", null: false
    t.bigint "away_team_id", null: false
    t.boolean "completed", default: false
    t.string "season_id"
    t.string "season_name"
    t.datetime "game_date_iso_8601"
    t.integer "home_goals"
    t.integer "away_goals"
    t.integer "home_shots"
    t.integer "away_shots"
    t.decimal "home_power_play_percentage", precision: 8, scale: 2
    t.decimal "away_power_play_percentage", precision: 8, scale: 2
    t.decimal "home_penalty_kill_percentage", precision: 8, scale: 2
    t.decimal "away_penalty_kill_percentage", precision: 8, scale: 2
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["away_team_id"], name: "index_whl_team_stats_on_away_team_id"
    t.index ["game_id"], name: "index_whl_team_stats_on_game_id", unique: true
    t.index ["home_team_id"], name: "index_whl_team_stats_on_home_team_id"
  end

  create_table "whl_teams", force: :cascade do |t|
    t.string "name", null: false
    t.string "hockeytech_id", null: false
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

  add_foreign_key "whl_games", "whl_teams", column: "away_team_id"
  add_foreign_key "whl_games", "whl_teams", column: "home_team_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "actual_winner_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "away_team_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "home_team_id"
  add_foreign_key "whl_prediction_records", "whl_teams", column: "predicted_winner_id"
  add_foreign_key "whl_rolling_averages", "whl_teams"
  add_foreign_key "whl_team_stats", "whl_teams", column: "away_team_id"
  add_foreign_key "whl_team_stats", "whl_teams", column: "home_team_id"
end
