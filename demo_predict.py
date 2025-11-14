"""
Quick demo: Load model and make predictions without Spark (for local testing)
"""

import pickle
import pandas as pd
import requests

# 1. Fetch campaign config
print("=" * 60)
print("DEMO: Simple Prediction Pipeline (without Spark)")
print("=" * 60)

print("\n📋 Fetching campaign config from API...")
response = requests.get("http://localhost:8000/campaigns/1")
config = response.json()
print(f"✅ Campaign: {config['name']}")
print(f"   Model: {config['model_path']}")
print(f"   Filter: {config['audience_filter']}")

# 2. Load model
print(f"\n🤖 Loading model...")
with open(config['model_path'], 'rb') as f:
    artifact = pickle.load(f)

model = artifact['model']
features = artifact['features']
print(f"✅ Model loaded: {artifact['metadata']['model_type']}")
print(f"   Accuracy: {artifact['metadata']['accuracy']:.3f}")

# 3. Load user data
print(f"\n👥 Loading user data...")
df = pd.read_parquet('data/sample_users.parquet')
print(f"✅ Loaded {len(df)} users")

# 4. Apply audience filter
print(f"\n🎯 Applying filter: {config['audience_filter']}")
# Apply filter using pandas
df_filtered = df[(df['country'] == 'US') & (df['age'] > 25)]
print(f"✅ Filtered to {len(df_filtered)} users ({len(df_filtered)/len(df)*100:.1f}%)")

# 5. Generate predictions
print(f"\n🔮 Generating predictions...")
X = df_filtered[features]
predictions = model.predict_proba(X)[:, 1]
df_filtered['prediction_score'] = predictions

# 6. Save results
output_file = 'output/predictions_campaign_1_demo.parquet'
df_filtered[['user_id', 'prediction_score'] + features].to_parquet(output_file)

print(f"\n💾 Saved predictions to: {output_file}")
print(f"\n📊 RESULTS:")
print(f"   Total predictions: {len(df_filtered):,}")
print(f"   Avg score: {predictions.mean():.3f}")
print(f"   Min score: {predictions.min():.3f}")
print(f"   Max score: {predictions.max():.3f}")

print(f"\n📋 Top 10 users to target:")
print(df_filtered[['user_id', 'prediction_score', 'country', 'age']].nlargest(10, 'prediction_score').to_string(index=False))

print("\n" + "=" * 60)
print("✅ DEMO COMPLETE!")
print("=" * 60)
