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

ActiveRecord::Schema[8.0].define(version: 2025_06_01_033437) do
  create_table "whl_prediction_records", force: :cascade do |t|
    t.integer "game_id"
    t.integer "k_value"
    t.float "home_prob"
    t.float "away_prob"
    t.boolean "correct"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.string "home_team"
    t.string "away_team"
  end

  create_table "whl_team_stats", force: :cascade do |t|
    t.integer "game_id"
    t.string "home_name"
    t.string "away_name"
    t.integer "home_goals"
    t.integer "away_goals"
    t.float "home_ppp"
    t.float "away_ppp"
    t.integer "home_sog"
    t.integer "away_sog"
    t.float "home_fowp"
    t.float "away_fowp"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end
end
