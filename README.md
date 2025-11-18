# Hybrid Crop Recommendation System

A hybrid crop recommendation system that combines rule-based expert knowledge (40%) with machine learning (60%) to recommend optimal crops for Filipino farmers based on soil conditions, location climate, and historical performance data.

## Features

- **Rule-Based Scoring (40%)**: Expert knowledge-based scoring using NPK matching, pH validation, climate suitability, and soil type matching
- **Trained ML Model (60%)**: Regression model trained on historical yield data to predict suitability scores
- **Hybrid Recommendation Engine**: Combines all three scoring methods for optimal recommendations
- **REST API**: FastAPI endpoint for easy integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the ML model (run notebooks in order):
   - `01_data_exploration.ipynb`
   - `02_feature_engineering.ipynb`
   - `03_model_training.ipynb`
   - `04_model_evaluation.ipynb`

3. Start the API server:
```bash
uvicorn app.main:app --reload
```

## API Usage

### POST /recommend

Request body:
```json
{
  "province": "Cagayan",
  "municipality": "Tuguegarao",
  "nitrogen": "Medium",
  "phosphorus": "High",
  "potassium": "Medium",
  "ph_min": 6.0,
  "ph_max": 6.8,
  "soil_type": "Loam"
}
```

Response: Top 10 crop recommendations with detailed scores, risks, and explanations.

## Development

The system uses a two-tier scoring approach:
1. **Rule-Based (50%)**: Expert rules for NPK, pH, climate, and soil matching (6 components)
2. **ML Model (50%)**: Trained regression model predicting suitability scores using 9 features (includes historical yield, season alignment, regional success)

Final hybrid score = (Rule × 0.4) + (ML Model × 0.6)

