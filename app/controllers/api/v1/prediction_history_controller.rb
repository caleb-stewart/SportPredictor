module Api
  module V1
    class PredictionHistoryController < ApplicationController

      def history
        
        predictions = WhlPredictionRecord.all.order(created_at: :desc).limit(100)
        # puts "PREDICTIONS", predictions.to_json

        # Format the predictions into a structure that is used by the frontend
        formatted_predictions = format_predictions(predictions)
        
        # puts "Formatted Predictions: ", formatted_predictions.to_json
        # Return the formatted predictions as JSON to frontend
        render json: formatted_predictions

      end # history()

      private

      # This method formats the predictions into a structure that is used by the frontend.
      def format_predictions(predictions)

        # Returns in this following format:
        # [
        #   {
        #     game_id: Integer,                // Unique ID for the game (e.g., 1022064)
        #     home_team: String,              // Name of the home team (e.g., "Spokane Chiefs")
        #     away_team: String,              // Name of the away team (e.g., "Medicine Hat Tigers")
        #     k_values: [                     // Array of prediction results by rolling window (K-value)
        #       {
        #         k_value: "5" | "10" | "15", // Size of the rolling average window as a string
        #         home_prob: Number,          // Probability home team wins (e.g., 0.3526)
        #         away_prob: Number,          // Probability away team wins (e.g., 0.6473)
        #         correct: Boolean | null     // Whether the prediction was correct (true/false), or null if unknown
        #       },
        #       ...
        #     ]
        #   },
        #   ...
        # ]

        # Creates groups by game_id
        grouped = predictions.group_by(&:game_id)

        puts "Grouped Predictions: ", grouped.to_json

        # Loop through each group (game_id) and format the predictions
        formatted_predictions = grouped.map do |game_id, records|

          # Get the k_values and their probabilities from the predictions previously made
          k_values = records.map do |k_prediction|

            # puts "K Prediction: ", k_prediction.to_json
            # Determine the key based on the k_value
            key = case k_prediction.k_value
            when 5 then '5'
            when 10 then '10'
            when 15 then '15'
            end # case

            # Check if our prediction was correct
            correct = correct_prediction?(k_prediction)
            

            # Return the formatted k_value prediction
            {
              k_value: key,
              home_prob: k_prediction.home_prob,
              away_prob: k_prediction.away_prob,
              correct: correct
            }
          end # records.map

          # Return the formatted prediction for this game
          {
            game_id: game_id,
            home_team: records.first.home_team,
            away_team: records.first.away_team,
            k_values: k_values # this was the array of k_values made from above
          }
        end # grouped.map

      end # format_predictions()

      def correct_prediction?(k_prediction)

        # Check the prediction record to see if our prediction was correct
        WhlPredictionRecord.find_by(game_id: k_prediction.game_id, k_value: k_prediction.k_value).correct

      end # correct_prediction?()

    end # class
  end # module V1
end # module Api