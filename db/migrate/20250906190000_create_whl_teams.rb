class CreateWhlTeams < ActiveRecord::Migration[7.1]
  def change
    create_table :whl_teams do |t|
      t.string :name, null: false
      t.string :hockeytech_id, null: false
      t.string :city
      t.string :team_name
      t.string :conference
      t.string :division
      t.boolean :active, default: true
      t.timestamps
    end
    add_index :whl_teams, :hockeytech_id, unique: true
    add_index :whl_teams, :name, unique: true
  end
end
