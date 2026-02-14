require "rake"

class WhlDailyModelRefreshJob < ApplicationJob
  queue_as :default

  def perform(*_args)
    Rails.application.load_tasks unless Rake::Task.task_defined?("predictor_v2:daily_pipeline")
    Rake::Task["predictor_v2:daily_pipeline"].reenable
    Rake::Task["predictor_v2:daily_pipeline"].invoke
  end
end
