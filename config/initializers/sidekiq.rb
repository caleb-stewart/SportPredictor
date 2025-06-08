require "sidekiq"
require "sidekiq-scheduler"

Sidekiq.configure_server do |config|
  config.on(:startup) do
    sidekiq_schedule_file = Rails.root.join("config", "sidekiq.yml")
    puts "Loading Sidekiq schedule from:", sidekiq_schedule_file
    Sidekiq.schedule = YAML.load_file(File.expand_path(sidekiq_schedule_file))[:scheduler][:schedule]
    SidekiqScheduler::Scheduler.instance.reload_schedule!
  end
end
