class CreateWhlTeamStats < ActiveRecord::Migration[8.0]
  def change
    create_table :whl_team_stats do |t|
      t.integer :game_id
      t.string :home_name
      t.string :away_name
      t.integer :home_goals
      t.integer :away_goals
      t.float :home_ppp
      t.float :away_ppp
      t.integer :home_sog
      t.integer :away_sog
      t.float :home_fowp
      t.float :away_fowp

      t.timestamps
    end
  end
end
