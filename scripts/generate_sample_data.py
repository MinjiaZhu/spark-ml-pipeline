"""
Generate sample user data for testing the Spark prediction job.

Creates a parquet file with 1000 users containing the same features
the model was trained on.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of sample users
N_USERS = 1000

# Feature names (must match training data)
feature_names = [
    'recency',                    # Days since last purchase
    'frequency',                  # Number of purchases
    'monetary',                   # Total spend
    'engagement_score',           # User engagement metric
    'days_since_last_purchase'    # Redundant with recency, but model expects it
]

# Generate synthetic user data
data = {
    'user_id': [f'user_{i:05d}' for i in range(N_USERS)],
    'recency': np.random.exponential(scale=30, size=N_USERS),
    'frequency': np.random.poisson(lam=5, size=N_USERS),
    'monetary': np.random.lognormal(mean=5, sigma=1, size=N_USERS),
    'engagement_score': np.random.uniform(0, 1, size=N_USERS),
    'days_since_last_purchase': np.random.exponential(scale=30, size=N_USERS),
}

# Add some categorical features for filtering
data['country'] = np.random.choice(['US', 'UK', 'CA', 'AU'], size=N_USERS)
data['age'] = np.random.randint(18, 70, size=N_USERS)
data['is_premium'] = np.random.choice([True, False], size=N_USERS, p=[0.3, 0.7])

# Create DataFrame
df = pd.DataFrame(data)

# Output path
output_path = Path(__file__).parent.parent / 'data' / 'sample_users.parquet'

# Save as parquet
df.to_parquet(output_path, index=False)

print(f"✅ Generated {N_USERS} sample users")
print(f"📁 Saved to: {output_path}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
