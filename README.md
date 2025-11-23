# CropTap ML v3: Hybrid Crop Recommendation System

**Version 2.0.0**

This project implements a hybrid crop recommendation system for Filipino farmers, combining rule-based logic with a machine learning model. It provides optimal crop recommendations via a FastAPI-based REST API, considering soil conditions, climate, and historical data.

The recommendation engine takes farmer inputs (province, municipality, soil NPK levels, pH, and soil type) and outputs a ranked list of the top 10 most suitable crops. Each recommendation includes a hybrid score (40% rule-based, 60% ML model), confidence, expected yield, potential risks, and fertilizer recommendations.

## Key Features

- **Hybrid Scoring**: Combines rule-based logic (40%) with RandomForest ML model (60%)
- **Comprehensive Recommendations**: Includes expected yield, risks, planting season, and fertilizer advice
- **Climate-Aware**: Uses real-time climate data for accurate predictions
- **Historical Analysis**: Leverages historical crop performance data
- **REST API**: Easy-to-use FastAPI endpoints with interactive documentation

## Getting Started

Follow these steps to set up, train the model, and run the API server.

### 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

Before starting the API, the machine learning model must be trained. Navigate to the `notebooks/` directory and execute the following Jupyter notebooks in order:

1. `01_data_exploration.ipynb` - Explore and understand the dataset
2. `02_feature_engineering.ipynb` - Create features for the ML model
3. `03_model_training.ipynb` - Train and compare different models
4. `04_model_evaluation.ipynb` - Evaluate model performance
5. `05_model_deep_dive.ipynb` - Comprehensive model analysis and feature importance

This process will generate the `crop_suitability_model.pkl` file in the `models/` directory, which is essential for the API.

### 3. Run the API Server

Once the model is trained and the `crop_suitability_model.pkl` file exists, you can start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

## API Endpoints

### POST /recommend
Get crop recommendations based on soil and location data.

**Request Body:**
```json
{
  "province": "LAGUNA",
  "municipality": "Bay",
  "nitrogen": "High",
  "phosphorus": "Medium",
  "potassium": "High",
  "ph_min": 6.0,
  "ph_max": 7.0,
  "soil_type": "Clay Loam"
}
```

**Response Example:**
```json
{
  "location": {
    "province": "LAGUNA",
    "municipality": "Bay"
  },
  "climate_summary": {
    "avg_temperature": 27.5,
    "avg_rainfall": 180.2,
    "avg_humidity": 78.5
  },
  "soil_summary": {
    "nitrogen": "High",
    "phosphorus": "Medium",
    "potassium": "High",
    "ph_range": "6.0-7.0",
    "soil_type": "Clay Loam"
  },
  "recommendations": [
    {
      "rank": 1,
      "crop_name": "Rice",
      "category": "Cereal",
      "hybrid_score": 87.45,
      "rule_score": 85.0,
      "ml_model_score": 89.0,
      "confidence": 96.0,
      "expected_yield": "4.5 tons/ha",
      "risks": [],
      "planting_season": "May-June, October-November",
      "days_to_harvest": "120-150 days",
      "fertilizer_recommendation": "Apply 14-14-14 fertilizer at planting",
      "why_recommended": "Excellent NPK match, ideal pH range, suitable temperature conditions, proven success in LAGUNA with avg 4.2 tons/ha over 5 years.",
      "badge": "Best Match",
      "rule_breakdown": {
        "npk": 28.0,
        "ph": 20.0,
        "temperature": 18.5,
        "rainfall": 12.0,
        "humidity": 4.5,
        "soil_type": 2.0
      },
      "ml_breakdown": {
        "npk_match": 0.95,
        "ph_proximity": 1.0,
        "temp_suitability": 0.92,
        "rainfall_suitability": 0.85,
        "humidity_suitability": 0.78,
        "soil_match": 1.0,
        "historical_yield": 0.88,
        "season_alignment": 1.0,
        "regional_success": 0.91
      }
    }
  ],
  "insights": {
    "best_season": "May-June for optimal planting",
    "soil_amendments": "Soil conditions are suitable for most crops",
    "diversification_tip": "Consider mixing Rice with Corn for crop rotation"
  }
}
```

### GET /crops
Get a list of all available crops in the database.

**Response Example:**
```json
{
  "crops": [
    {
      "Crop_Name": "Rice",
      "Category": "Cereal",
      "Nitrogen": "High",
      "Phosphorus": "Medium",
      "Potassium": "High",
      "pH_min": 5.5,
      "pH_max": 7.0,
      "Temp_min": 20.0,
      "Temp_max": 35.0,
      "Rainfall_min": 1000.0,
      "Rainfall_max": 2500.0,
      "Humidity_min": 70.0,
      "Humidity_max": 90.0,
      "Soil_Type": "Clay Loam",
      "Planting Period": "May-June, October-November",
      "Days to Harvest": "120-150 days",
      "nutrient_notes": "Apply 14-14-14 fertilizer at planting"
    },
    {
      "Crop_Name": "Corn",
      "Category": "Cereal",
      "Nitrogen": "High",
      "Phosphorus": "High",
      "Potassium": "Medium"
    }
  ]
}
```

### GET /climate
Get climate data for a specific location.

**Query Parameters:**
- `province` (required): Province name (e.g., "LAGUNA")
- `municipality` (optional): Municipality name (e.g., "Bay")

**Example Request:**
```
GET /climate?province=LAGUNA&municipality=Bay
```

**Response Example:**
```json
{
  "climate": {
    "temperature": 27.5,
    "rainfall": 180.2,
    "humidity": 78.5,
    "month": 11
  }
}
```

### GET /health
Health check endpoint to verify API is running.

**Response Example:**
```json
{
  "status": "healthy",
  "message": "CropTap ML API is running",
  "version": "2.0.0",
  "model_loaded": true
}
```

## Project Structure

The project follows a modular and service-oriented architecture:

*   **`app/`**: Main application directory
    *   **`main.py`**: FastAPI application entry point
    *   **`models/`**: Pydantic schemas for API request/response validation
    *   **`services/`**: Business logic for recommendation engine, data loading, and scoring
        *   `data_loader.py`: Loads and manages crop and climate datasets
        *   `recommendation_engine.py`: Main recommendation engine
        *   `rule_based_scorer.py`: Rule-based scoring logic
        *   `feature_extractor.py`: ML feature extraction
        *   `ml_model.py`: ML model prediction service
        *   `yield_predictor.py`: Expected yield calculation
    *   **`utils/`**: Utility functions for data processing and validation
*   **`notebooks/`**: Jupyter notebooks for data analysis and model training
*   **`raw_datasets/`**: Raw CSV data files (117 crops, 94,830 historical records, 215,556 climate records)
*   **`models/`**: Trained machine learning models and metadata

## Technical Details

- **Model**: RandomForestRegressor with 100 estimators
- **Features**: 10 engineered features including NPK match, pH proximity, climate suitability, historical yield, season alignment, and regional success
- **Scoring**: Hybrid approach combining domain expertise with ML predictions
- **Data**: Comprehensive Philippine agricultural data including climate, soil, and historical performance
