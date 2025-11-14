Architecture Overview
┌─────────────────┐
│   FastAPI       │  ← Config Management API (CRUD)
│   Config API    │  
└────────┬────────┘
         │ stores/reads
         ▼
┌─────────────────┐
│   PostgreSQL    │  ← Campaign configurations
│   Config DB     │  
└─────────────────┘
                        
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Sample User    │────▶│  Spark Predict   │────▶│  Predictions    │
│  Features       │     │  Job (PySpark)   │     │  Output         │
│  (Parquet)      │     │  + CatBoost      │     │  (Parquet)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                ▲
                                │ loads
                        ┌───────┴────────┐
                        │ Model Artifact │
                        │ (.pkl file)    │
                        │  CatBoost      │
                        └────────────────┘
The Flow:

Config API: Marketer creates campaign config via FastAPI (which model, which audience)
Batch Job: Spark job reads config → loads users → loads model → scores users
Results: Predictions written to parquet (in prod: BigQuery)

---

# Spark ML Pipeline - MLE Learning Project

A production-style ML pipeline demonstrating:
- **FastAPI** for campaign configuration CRUD
- **CatBoost** model training (tree-based classifier)
- **PySpark** for distributed batch prediction
- **PostgreSQL** for config storage
- **Docker** for local development
- **Kubernetes/DataProc** deployment patterns

## Prerequisites

1. **Python 3.9+** - `python3 --version`
2. **Docker Desktop** - For PostgreSQL container
3. **Java 11+** - Required by PySpark: `java -version`

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
# Install Python packages
pip3 install -r requirements.txt
```

### 2. Start Infrastructure

```bash
# Start PostgreSQL with Docker
docker-compose up -d

# Verify it's running
docker ps
```

### 3. Train the Model

```bash
# Train CatBoost model and save to models/
python3 2_model_training/train_catboost_model.py
```

Expected output:
```
Model accuracy: 0.XXX
✅ Model saved to models/campaign_model_v1.pkl
Required features: ['recency', 'frequency', 'monetary', 'engagement_score', 'days_since_last_purchase']
```

### 4. Generate Sample Data

```bash
# Create sample users for testing
python3 scripts/generate_sample_data.py
```

This creates `data/sample_users.parquet` with 1000 synthetic users.

### 5. Start the Config API

```bash
# Navigate to API directory
cd 1_config_api

# Run FastAPI with uvicorn
python3 -m uvicorn main:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 6. Create a Campaign (New Terminal)

```bash
# Create a campaign configuration
curl -X POST http://localhost:8000/campaigns/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Black Friday Campaign",
    "model_path": "models/campaign_model_v1.pkl",
    "audience_filter": "country='\''US'\'' AND age > 25",
    "features": ["recency", "frequency", "monetary", "engagement_score", "days_since_last_purchase"]
  }'
```

Response:
```json
{
  "id": 1,
  "name": "Black Friday Campaign",
  "model_path": "models/campaign_model_v1.pkl",
  "audience_filter": "country='US' AND age > 25",
  "features": [...],
  "is_active": true,
  "created_at": "2024-11-13T..."
}
```

### 7. Run Spark Prediction Job

```bash
# From project root
./3_predict_job/run_local.sh 1

# Or manually with spark-submit
spark-submit \
  --master local[*] \
  --driver-memory 2g \
  3_predict_job/predict_task.py \
  --campaign-id 1 \
  --config-api-url http://localhost:8000
```

Expected output:
```
🚀 SPARK PREDICTION JOB STARTING
📡 Fetching campaign config from: http://localhost:8000/campaigns/1
✅ Loaded campaign: Black Friday Campaign
📦 Loading model from: models/campaign_model_v1.pkl
📊 Reading user data from: data/sample_users.parquet
✅ Loaded 1,000 users
🎯 Applying audience filter: country='US' AND age > 25
✅ Filtered to 312 users (31.2%)
🤖 Generating predictions...
💾 Writing predictions to: output/predictions_campaign_1.parquet
✅ JOB COMPLETED SUCCESSFULLY
```

### 8. View Results

```bash
# Check output
ls -lh output/

# Read predictions with pandas
python3 -c "
import pandas as pd
df = pd.read_parquet('output/predictions_campaign_1.parquet')
print(df.head(10))
print(f'\nTotal predictions: {len(df)}')
print(f'Average score: {df.prediction_score.mean():.3f}')
"
```

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/campaigns/` | Create campaign |
| GET | `/campaigns/{id}` | Get campaign by ID |
| GET | `/campaigns/` | List all campaigns |
| PUT | `/campaigns/{id}` | Update campaign |
| DELETE | `/campaigns/{id}` | Delete campaign |
| GET | `/docs` | Interactive API docs |

## Project Structure

```
spark-ml-pipeline/
├── 1_config_api/           # FastAPI application
│   ├── main.py            # API routes (CRUD)
│   ├── models.py          # Pydantic schemas
│   └── database.py        # SQLAlchemy ORM
├── 2_model_training/       # Model training
│   └── train_catboost_model.py
├── 3_predict_job/          # Spark prediction job
│   ├── predict_task.py    # Main Spark job
│   ├── data_processing.py # Ibis example
│   └── run_local.sh       # Wrapper script
├── 4_kubernetes/           # Deployment configs
│   └── spark-job.yaml     # K8s/DataProc templates
├── scripts/                # Utilities
│   └── generate_sample_data.py
├── data/                   # User data (parquet)
├── models/                 # Trained models (.pkl)
├── output/                 # Predictions (parquet)
├── requirements.txt
└── docker-compose.yml
```

## Technologies Demonstrated

### Backend & API
- **FastAPI** - Modern async Python web framework
- **SQLAlchemy** - ORM for PostgreSQL
- **Pydantic** - Data validation and serialization

### Machine Learning
- **CatBoost** - Gradient boosting classifier
- **scikit-learn** - Model evaluation and data splitting
- **Pickle** - Model serialization pattern

### Data Processing
- **PySpark** - Distributed data processing
- **Ibis** - Portable dataframe library (see `data_processing.py`)
- **Pandas** - For Pandas UDFs in Spark

### Infrastructure
- **PostgreSQL** - Campaign config database
- **Docker Compose** - Local orchestration
- **Kubernetes** - Production deployment (see `4_kubernetes/`)

## Deployment

### Local Testing
```bash
docker-compose up -d
./scripts/demo.sh 
```

### Kubernetes
```bash
# Deploy with spark-on-k8s-operator
kubectl apply -f 4_kubernetes/spark-job.yaml
```

## Cleanup

```bash
# Stop PostgreSQL
docker-compose down

# Remove generated data
rm -rf data/*.parquet output/*.parquet models/*.pkl
```