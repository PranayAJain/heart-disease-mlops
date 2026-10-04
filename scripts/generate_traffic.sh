#!/usr/bin/env bash
# Generate mixed traffic for the monitoring demo (valid high/low risk + invalid requests).
# Usage: scripts/generate_traffic.sh [base_url] [seconds]
BASE="${1:-http://localhost:8000}"; DURATION="${2:-120}"
END=$((SECONDS + DURATION))
while [ $SECONDS -lt $END ]; do
  for f in high_risk low_risk low_risk invalid; do
    curl -s -o /dev/null -X POST "$BASE/predict" -H "Content-Type: application/json" \
         -d @samples/patient_$f.json
  done
  curl -s -o /dev/null "$BASE/health"
  sleep 0.5
done
echo "traffic generation finished"
