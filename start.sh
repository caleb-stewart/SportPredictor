#!/bin/bash

# Function to kill all child processes when the script exits
cleanup() {
  echo "[BASH] Stopping all servers..."
  # kill all background jobs started by this script
  kill $(jobs -p) 2>/dev/null
}

# Trap EXIT signal (script ending) to run cleanup
trap cleanup EXIT

echo "[BASH] Starting Rails server on port 3141..."
rails s -p 3141 &

echo "[BASH] Starting Flask server on port 2718..."
python3 PredictorFlask/app.py &

echo "[BASH] Starting Vue app on port 5173..."
cd ../sport-predictor-vue
npm run dev &

# Wait for all background jobs to finish (so the script doesn't exit immediately)
wait
