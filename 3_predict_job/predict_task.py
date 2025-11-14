"""
Spark Prediction Job - Batch scoring of users for marketing campaigns.

This job:
1. Fetches campaign configuration from FastAPI
2. Loads a trained ML model (CatBoost)
3. Reads user data from parquet
4. Applies audience filters
5. Generates predictions
6. Writes results to parquet

Usage:
    spark-submit predict_task.py --campaign-id 1 --config-api-url http://localhost:8000
"""

import argparse
import pickle
import requests
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql.types import DoubleType, StructType, StructField, StringType
import pandas as pd


def fetch_campaign_config(api_url: str, campaign_id: int) -> dict:
    """Fetch campaign configuration from FastAPI."""
    url = f"{api_url}/campaigns/{campaign_id}"
    print(f"=á Fetching campaign config from: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        config = response.json()
        print(f" Loaded campaign: {config['name']}")
        print(f"   Model: {config['model_path']}")
        print(f"   Features: {config['features']}")
        print(f"   Filter: {config['audience_filter']}")
        return config
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch campaign config: {e}")


def load_model(model_path: str) -> dict:
    """Load pickled model artifact."""
    # Handle relative paths from project root
    if not model_path.startswith('/'):
        project_root = Path(__file__).parent.parent
        model_path = project_root / model_path

    print(f"=æ Loading model from: {model_path}")

    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)

    print(f" Model loaded successfully")
    print(f"   Type: {model_artifact['metadata']['model_type']}")
    print(f"   Version: {model_artifact['metadata']['version']}")
    print(f"   Accuracy: {model_artifact['metadata']['accuracy']:.3f}")

    return model_artifact


def create_prediction_udf(model_artifact: dict, feature_names: list):
    """
    Create a Pandas UDF for batch prediction.

    This allows us to use the CatBoost model (which isn't Spark-native)
    by applying it in batches using pandas DataFrames.
    """
    model = model_artifact['model']

    @pandas_udf(DoubleType())
    def predict_udf(*cols):
        """Pandas UDF that applies the model to batches of data."""
        # Convert input columns to DataFrame
        df = pd.DataFrame({name: col for name, col in zip(feature_names, cols)})

        # Get predictions (probability of positive class)
        predictions = model.predict_proba(df)[:, 1]

        return pd.Series(predictions)

    return predict_udf


def parse_audience_filter(filter_str: str) -> str:
    """
    Convert SQL-like filter to Spark SQL format.

    For now, we'll use it directly since it should already be valid SQL.
    In production, you'd want to validate and sanitize this.
    """
    # Simple validation - ensure no dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER']
    if any(keyword in filter_str.upper() for keyword in dangerous_keywords):
        raise ValueError(f"Invalid filter contains dangerous SQL keyword: {filter_str}")

    return filter_str


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run batch prediction for a campaign')
    parser.add_argument('--campaign-id', type=int, required=True,
                       help='Campaign ID from config API')
    parser.add_argument('--config-api-url', type=str, default='http://localhost:8000',
                       help='Base URL for config API')
    parser.add_argument('--input-data', type=str, default='data/sample_users.parquet',
                       help='Input parquet file with user data')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for predictions')

    args = parser.parse_args()

    print("=" * 80)
    print("=€ SPARK PREDICTION JOB STARTING")
    print("=" * 80)
    print(f"Campaign ID: {args.campaign_id}")
    print(f"Config API: {args.config_api_url}")
    print(f"Input Data: {args.input_data}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 80)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"CampaignPrediction-{args.campaign_id}") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    try:
        # Step 1: Fetch campaign configuration
        config = fetch_campaign_config(args.config_api_url, args.campaign_id)

        # Step 2: Load model
        model_artifact = load_model(config['model_path'])
        feature_names = config['features']

        # Step 3: Read user data
        print(f"\n=Ê Reading user data from: {args.input_data}")
        df_users = spark.read.parquet(args.input_data)
        print(f" Loaded {df_users.count():,} users")
        print(f"   Columns: {df_users.columns}")

        # Step 4: Apply audience filter
        if config['audience_filter']:
            audience_filter = parse_audience_filter(config['audience_filter'])
            print(f"\n<¯ Applying audience filter: {audience_filter}")
            df_filtered = df_users.filter(audience_filter)
            filtered_count = df_filtered.count()
            print(f" Filtered to {filtered_count:,} users ({filtered_count/df_users.count()*100:.1f}%)")
        else:
            df_filtered = df_users
            print("\n<¯ No audience filter applied")

        # Step 5: Validate features exist
        missing_features = set(feature_names) - set(df_filtered.columns)
        if missing_features:
            raise ValueError(f"Missing required features in data: {missing_features}")

        # Step 6: Create prediction UDF
        predict_udf = create_prediction_udf(model_artifact, feature_names)

        # Step 7: Generate predictions
        print(f"\n> Generating predictions...")
        feature_cols = [col(f) for f in feature_names]
        df_predictions = df_filtered.withColumn(
            'prediction_score',
            predict_udf(*feature_cols)
        )

        # Step 8: Select output columns
        output_cols = ['user_id', 'prediction_score'] + feature_names
        df_output = df_predictions.select(*output_cols)

        # Step 9: Write predictions to parquet
        output_path = f"{args.output_dir}/predictions_campaign_{args.campaign_id}.parquet"
        print(f"\n=¾ Writing predictions to: {output_path}")

        df_output.write.mode('overwrite').parquet(output_path)

        # Step 10: Show summary statistics
        print(f"\n=È PREDICTION SUMMARY")
        print("=" * 80)
        df_output.select('prediction_score').summary().show()

        print("\n=Ê Sample predictions:")
        df_output.orderBy(col('prediction_score').desc()).show(10, truncate=False)

        print("\n" + "=" * 80)
        print(" JOB COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"=Á Output: {output_path}")
        print(f"=Ê Predictions: {df_output.count():,} users")

    except Exception as e:
        print(f"\nL ERROR: {e}")
        raise

    finally:
        spark.stop()


if __name__ == '__main__':
    main()
