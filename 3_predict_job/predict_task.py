"""
Spark batch prediction job

In production:
- Runs on managed Spark cluster
- Reads from db
- Writes results back to db
- Orchestrated by Airflow

In this demo:
- Runs locally with PySpark
- Reads from Parquet files
- smaller scale
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, struct, to_json
import pickle
import requests
import pandas as pd
from typing import List
import polars as pl  # Using Polars for preprocessing
import ibis  # Alternative to show you learned both

def load_campaign_config(campaign_id: int):
    """
    Fetch campaign configuration from the Config API.
    This is how the predict job knows which model to use.
    """
    response = requests.get(f"http://localhost:8000/campaigns/{campaign_id}")
    if response.status_code != 200:
        raise ValueError(f"Campaign {campaign_id} not found")
    return response.json()

def load_model_artifact(model_path: str):
    """
    Load pickled model from disk.
    In production: would load from GCS.
    """
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact['model'], artifact['features'], artifact['metadata']

def preprocess_with_polars(df_pandas: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Use Polars for fast preprocessing (showcasing you learned it).
    
    Polars is 10x faster than Pandas for large datasets.
    Perfect for data preparation before Spark processing.
    """
    # Convert to Polars DataFrame
    df_pl = pl.from_pandas(df_pandas)
    
    # Example preprocessing: filter nulls, compute derived features
    df_pl = df_pl.filter(
        pl.col('recency').is_not_null()
    ).with_columns([
        (pl.col('frequency') / pl.col('recency')).alias('engagement_velocity')
    ])
    
    # Select only required features
    df_pl = df_pl.select(required_features)
    
    return df_pl.to_pandas()

def run_predictions(campaign_id: int, spark: SparkSession):
    """
    Main prediction pipeline - the heart of OfferFit's system.
    """
    print(f"\n🚀 Starting prediction job for campaign {campaign_id}")
    
    # Step 1: Load campaign configuration from API
    print("📋 Loading campaign config...")
    config = load_campaign_config(campaign_id)
    print(f"   Campaign: {config['name']}")
    print(f"   Model: {config['model_path']}")
    
    # Step 2: Load the ML model artifact
    print("🤖 Loading model artifact...")
    model, required_features, metadata = load_model_artifact(config['model_path'])
    print(f"   Model type: {metadata['model_type']}")
    print(f"   Required features: {required_features}")
    
    # Step 3: Load user data (in production: from BigQuery)
    print("👥 Loading user features...")
    users_df = spark.read.parquet("../data/sample_users.parquet")
    print(f"   Loaded {users_df.count()} users")
    
    # Step 4: Apply audience filter from config
    if config.get('audience_filter'):
        print(f"🎯 Applying audience filter: {config['audience_filter']}")
        users_df = users_df.filter(config['audience_filter'])
        print(f"   Filtered to {users_df.count()} users")
    
    # Step 5: Preprocess with Polars (showcasing it)
    print("⚙️  Preprocessing with Polars...")
    users_pandas = users_df.toPandas()
    users_processed = preprocess_with_polars(users_pandas, required_features)
    
    # Step 6: Generate predictions using the model
    print("🔮 Generating predictions...")
    predictions = model.predict_proba(users_processed)[:, 1]  # Probability of engagement
    
    # Step 7: Combine with user IDs and write results
    print("💾 Writing predictions...")
    users_pandas['prediction_score'] = predictions
    users_pandas['campaign_id'] = campaign_id
    users_pandas['model_version'] = metadata['version']
    
    # Convert back to Spark and write (in production: to BigQuery)
    predictions_df = spark.createDataFrame(users_pandas[['user_id', 'prediction_score', 'campaign_id', 'model_version']])
    predictions_df.write.mode('overwrite').parquet(f"../data/predictions/campaign_{campaign_id}")
    
    print(f"✅ Predictions complete! Results saved to data/predictions/campaign_{campaign_id}")
    print(f"   Average prediction score: {predictions.mean():.3f}")
    
    return predictions_df

if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("OfferFit-Style Predict Job") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    # Run prediction for campaign ID 1
    campaign_id = 1
    
    try:
        results = run_predictions(campaign_id, spark)
        results.show(10)
    finally:
        spark.stop()