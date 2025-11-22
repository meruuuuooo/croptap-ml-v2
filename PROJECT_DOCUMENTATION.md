# CropTap ML v3 Project Documentation

## Project Overview

This project is a hybrid crop recommendation system designed for Filipino farmers. It combines a rule-based approach with a machine learning model to provide optimal crop recommendations based on soil conditions, climate, and historical performance data. The system is delivered as a REST API using FastAPI.

The core of the system is a recommendation engine that takes farmer inputs (province, municipality, soil NPK levels, pH, and soil type) and returns a ranked list of the top 10 most suitable crops. Each recommendation is enriched with a hybrid score, a rule-based score, an ML model score, a confidence level, expected yield, potential risks, and other useful information.

The hybrid scoring is weighted, with 40% of the score coming from the rule-based system and 60% from the machine learning model.

## File and Directory Structure

The project is organized into the following directories and files:

-   **`/app`**: Contains the main FastAPI application code.
    -   **`main.py`**: The entry point of the FastAPI application. It defines the API endpoints.
    -   **`/models`**: Contains Pydantic schemas for API request and response validation.
        -   **`schemas.py`**: Defines the `RecommendationRequest` and `RecommendationResponse` models.
    -   **`/services`**: Contains the business logic for the recommendation engine.
        -   **`recommendation_engine.py`**: Orchestrates the recommendation process.
        -   **`rule_based_scorer.py`**: Implements the rule-based scoring logic.
        -   **`ml_model.py`**: Loads and uses the trained machine learning model.
        -   **`feature_extractor.py`**: Extracts features for the ML model.
        -   **`data_loader.py`**: Loads data from CSV files.
        -   **`yield_predictor.py`**: Predicts the expected yield of a crop.
    -   **`/utils`**: Contains utility functions.
-   **`/notebooks`**: Contains Jupyter notebooks for the machine learning pipeline.
    -   **`01_data_exploration.ipynb`**: Explores the raw datasets.
    -   **`02_feature_engineering.ipynb`**: Creates the training dataset and engineers features.
    -   **`03_model_training.ipynb`**: Trains and compares machine learning models.
    -   **`04_model_evaluation.ipynb`**: Evaluates the trained model.
-   **`/raw_datasets`**: Contains the raw data used for training and recommendations.
-   **`/models`**: Contains the trained machine learning model (`crop_suitability_model.pkl`).
-   **`requirements.txt`**: Lists the Python dependencies for the project.
-   **`GEMINI.md`**: A brief overview of the project.
-   **`PROJECT_DOCUMENTATION.md`**: This file.

## Machine Learning Pipeline

The machine learning pipeline is a series of steps that take raw data and transform it into a trained model that can be used for predictions. The pipeline is implemented as a series of Jupyter notebooks in the `/notebooks` directory.

### 1. Data Exploration

-   **Notebook**: `01_data_exploration.ipynb`
-   **Purpose**: To understand the characteristics of the raw data, identify missing values, and uncover relationships between variables.
-   **Key Findings**:
    -   Crop yields are heavily right-skewed.
    -   Cassava and Sweet Potato are consistently high-yield crops.
    -   Data density varies significantly across provinces.

### 2. Feature Engineering

-   **Notebook**: `02_feature_engineering.ipynb`
-   **Purpose**: To create a clean, normalized, and feature-rich dataset for training the model.
-   **Process**:
    1.  **Data Merging**: The `historical_crop_performance.csv` is merged with `soil_test_data.csv` and `climate_data.csv`.
    2.  **Feature Extraction**: 9 core features are generated to quantify the suitability of a crop for a given set of inputs. These features include `npk_match`, `ph_proximity`, `temp_suitability`, `rainfall_suitability`, `humidity_suitability`, `soil_match`, `historical_yield`, `season_alignment`, and `regional_success`.
    3.  **Target Variable Creation**: A `suitability_score` is created by normalizing the `yield_per_ha` for a given crop against its 95th percentile yield. This makes the target robust to outliers.

### 3. Model Training

-   **Notebook**: `03_model_training.ipynb`
-   **Purpose**: To train and evaluate several regression models to find the best one for predicting the `suitability_score`.
-   **Models Trained**:
    -   `RandomForestRegressor`
    -   `XGBRegressor`
    -   `GradientBoostingRegressor`
-   **Best Model**: The `RandomForestRegressor` was chosen for its balance of accuracy and generalization.
-   **Output**: The trained model is saved to `models/crop_suitability_model.pkl`.

### 4. Model Evaluation

-   **Notebook**: `04_model_evaluation.ipynb`
-   **Purpose**: To formally evaluate the performance of the trained model.
-   **Metrics**:
    -   **R-squared (R²)**: Measures the proportion of the variance in the `suitability_score` that is predictable from the features.
    -   **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual `suitability_score`.
    -   **Root Mean Squared Error (RMSE)**: The standard deviation of the prediction errors.
-   **Interpretability**:
    -   **Feature Importance**: A plot is generated to show which features are most important to the model's predictions.
    -   **Prediction Analysis**: A scatter plot of predicted vs. actual scores is generated to visualize the model's performance.

## API Documentation

The API is built using FastAPI and can be accessed at `http://127.0.0.1:8000` when running locally. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

-   **`POST /recommend`**: The main endpoint for getting crop recommendations. It takes a `RecommendationRequest` and returns a `RecommendationResponse`.
-   **`GET /health`**: A health check endpoint that returns the status of the API and its components.
-   **`GET /`**: The root endpoint, which provides basic information about the API.

### Request Schema

The `POST /recommend` endpoint expects a JSON body with the following structure (`RecommendationRequest`):

```json
{
  "province": "string",
  "municipality": "string",
  "nitrogen": "Low" | "Medium" | "High",
  "phosphorus": "Low" | "Medium" | "High",
  "potassium": "Low" | "Medium" | "High",
  "ph_min": 0.0,
  "ph_max": 0.0,
  "soil_type": "string"
}
```

### Response Schema

The `POST /recommend` endpoint returns a JSON object with the following structure (`RecommendationResponse`):

```json
{
  "location": {
    "province": "string",
    "municipality": "string"
  },
  "climate_summary": {
    "avg_temperature": 0.0,
    "avg_rainfall": 0.0,
    "avg_humidity": 0.0
  },
  "soil_summary": {
    "nitrogen": "string",
    "phosphorus": "string",
    "potassium": "string",
    "ph_range": "string",
    "soil_type": "string"
  },
  "recommendations": [
    {
      "rank": 0,
      "crop_name": "string",
      "category": "string",
      "hybrid_score": 0.0,
      "rule_score": 0.0,
      "ml_model_score": 0.0,
      "confidence": 0.0,
      "badge": "string",
      "expected_yield": "string",
      "risks": [
        "string"
      ],
      "planting_season": "string",
      "days_to_harvest": "string",
      "fertilizer_recommendation": "string",
      "why_recommended": "string",
      "rule_breakdown": {
        "npk": 0.0,
        "ph": 0.0,
        "temperature": 0.0,
        "rainfall": 0.0,
        "humidity": 0.0,
        "soil_type": 0.0
      },
      "ml_breakdown": {
        "npk_match": 0.0,
        "ph_proximity": 0.0,
        "temp_suitability": 0.0,
        "rainfall_suitability": 0.0,
        "humidity_suitability": 0.0,
        "soil_match": 0.0,
        "historical_yield": 0.0,
        "season_alignment": 0.0,
        "regional_success": 0.0
      }
    }
  ],
  "insights": {
    "best_season": "string",
    "soil_amendments": "string",
    "diversification_tip": "string"
  }
}
```

## Core Logic Services

The business logic of the application is encapsulated in a set of services located in the `app/services` directory. Each service has a specific responsibility:

-   **`recommendation_engine.py`**: This is the main service that orchestrates the entire recommendation process. It takes the user's input and calls the other services to get the rule-based score, the ML model score, and other information. It then combines this information to generate the final recommendations.
-   **`rule_based_scorer.py`**: This service implements the rule-based scoring logic. It scores crops based on how well their requirements for NPK, pH, climate, and soil type match the user's input.
-   **`ml_model.py`**: This service is responsible for loading the trained `RandomForestRegressor` model and using it to predict a `suitability_score`.
-   **`feature_extractor.py`**: This service extracts the 9 core features that are used as input for the machine learning model.
-   **`data_loader.py`**: This service loads all the necessary data from the CSV files in the `raw_datasets` directory. It uses a singleton pattern to ensure that the data is only loaded once.
-   **`yield_predictor.py`**: This service predicts the expected yield of a crop based on historical data and the model's output.
-   **`ml_scorer.py`**: This file is dead code from a previous implementation and is not used.

This documentation provides a comprehensive overview of the CropTap ML v3 project.
