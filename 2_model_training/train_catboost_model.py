#very simple training just to make thing run

import pandas as pd
from catboost import CatBoostClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic user engagement data
X, y = make_classification(
    n_samples=10000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    random_state=42
)

feature_names = ['recency', 'frequency', 'monetary', 'engagement_score', 'days_since_last_purchase']
df = pd.DataFrame(X, columns=feature_names)
df['will_engage'] = y

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['will_engage'], test_size=0.2, random_state=42
)

# Train CatBoost model (tree-based, like OfferFit uses)
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False
)

model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")

# pickle file: picking is just serializing an object so we can save somewhere and unpack when need it
#saving actual instnace of a class - ML features?
with open('../models/campaign_model_v1.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'features': feature_names,
        'metadata': {
            'model_type': 'CatBoostClassifier',
            'version': 'v1',
            'accuracy': accuracy
        }
    }, f)

print("✅ Model saved to models/campaign_model_v1.pkl")
print(f"Required features: {feature_names}")