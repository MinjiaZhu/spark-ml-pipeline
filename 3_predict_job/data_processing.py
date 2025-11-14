"""
Alternative data processing using Ibis - just to show that I looked into both Polars and Ibis
"""
import ibis
from ibis import _

def preprocess_with_ibis(df_path: str):
    """
    Use Ibis for SQL-like data transformations.
    
    Ibis is great for:
    - Portable SQL-like code
    - Works across backends (DuckDB, BigQuery, Spark)
    - Your SQL skills translate directly
    """
    # Connect to data (using DuckDB backend for local demo)
    con = ibis.duckdb.connect()
    users = con.read_parquet(df_path, table_name='users')
    
    # SQL-like transformations
    processed = (
        users
        .filter(_.recency.notnull())
        .mutate(
            engagement_velocity=_.frequency / _.recency,
            high_value=_.monetary > 100
        )
        .select('user_id', 'recency', 'frequency', 'monetary', 'engagement_velocity')
    )
    
    return processed.to_pandas()

# Example usage
if __name__ == "__main__":
    df = preprocess_with_ibis('../data/sample_users.parquet')
    print(f"Processed {len(df)} users with Ibis")
    print(df.head())