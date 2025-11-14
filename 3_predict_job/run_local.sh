#!/bin/bash

# Wrapper script for running Spark prediction job locally
#
# Usage:
#   ./run_local.sh <campaign_id> [config_api_url]
#
# Examples:
#   ./run_local.sh 1
#   ./run_local.sh 1 http://localhost:8000

set -e  # Exit on error

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <campaign_id> [config_api_url]"
    echo ""
    echo "Examples:"
    echo "  $0 1"
    echo "  $0 1 http://localhost:8000"
    exit 1
fi

CAMPAIGN_ID=$1
CONFIG_API_URL=${2:-http://localhost:8000}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Running Spark Prediction Job"
echo "========================================"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Config API:  $CONFIG_API_URL"
echo "Project Root: $PROJECT_ROOT"
echo "========================================"

# Navigate to project root
cd "$PROJECT_ROOT"

# Run spark-submit
spark-submit \
    --master local[*] \
    --driver-memory 2g \
    --conf spark.sql.execution.arrow.pyspark.enabled=true \
    3_predict_job/predict_task.py \
    --campaign-id "$CAMPAIGN_ID" \
    --config-api-url "$CONFIG_API_URL" \
    --input-data "data/sample_users.parquet" \
    --output-dir "output"

echo ""
echo "========================================"
echo "Job completed! Check output/ directory"
echo "========================================"
