# CropTap ML Pipeline Documentation

This document provides a comprehensive, top-to-bottom overview of the machine learning pipeline for the CropTap recommendation system. It details each phase, from data ingestion to model deployment, reflecting the project's specific implementation.

## 1. Project Overview

The CropTap system is a **hybrid crop recommendation engine** designed for Filipino farmers. It provides optimal crop recommendations by combining two distinct methodologies:

1.  **Rule-Based System:** A static scoring system that matches farmer inputs against predefined optimal conditions for each crop.
2.  **Machine Learning Model:** A predictive model trained on historical performance and environmental data to forecast a "suitability score."

The final recommendation is a **hybrid score**, weighted with **40%** from the rule-based system and **60%** from the ML model. The system is delivered as a REST API built with **FastAPI**.

---

## 2. Phase 1: Data Ingestion

This phase deals with loading all necessary data from the local file system into the application's memory.

-   **Sources:** The pipeline ingests data from a collection of static CSV files stored in the `raw_datasets/` directory.
-   **Process:** The `app.services.data_loader.DataLoader` class is a singleton service responsible for loading all datasets into pandas DataFrames on application startup. This ensures data is read only once for efficiency.
-   **Key Datasets:**
    -   `historical_crop_performance.csv`: Contains historical data on crop yields (volume and area) for different locations and years.
    -   `soil_test_data.csv`: Provides real-world soil analysis information (N, P, K, pH) aggregated by province.
    -   `climate_data.csv`: Contains 5-year average climate data (temperature, rainfall, humidity) for various municipalities.
    -   `crop_requirements.csv`, `crop_npk_level_requirements.csv`, `crop_climate_requirements.csv`: These files contain the rule-based information on ideal conditions (pH, NPK levels, climate) for each crop.

---

## 3. Phase 2: Data Exploration

Before feature engineering, an exploratory data analysis (EDA) is performed in the `notebooks/01_data_exploration.ipynb` notebook to understand the data's characteristics.

-   **Objective:** To analyze data distributions, identify missing values, uncover relationships between variables, and understand historical yield patterns.
-   **Key Findings:**
    -   Crop yields are heavily right-skewed, with many low-yield records and a few high-yield outliers.
    -   Top-performing crops by average yield include Cassava and Sweet Potato.
    -   Data density varies significantly across provinces.
-   **Visualizations:** The notebook generates several plots for documentation and analysis, including:
    -   Distribution of crop yields (Histogram)
    -   Top 20 crops and provinces by average yield (Bar Charts)
    -   Average crop yield over time (Line Chart)
    -   Correlation matrix for production volume, area, and yield (Heatmap)
    -   Distribution of climate variables (Histograms)

---

## 4. Phase 3: Feature Engineering

This phase, executed in `notebooks/02_feature_engineering.ipynb`, focuses on creating a clean, feature-rich dataset suitable for model training.

-   **Data Merging:**
    1.  The `historical_crop_performance.csv` is merged with `soil_test_data.csv` using the province as the key.
    2.  The soil data is pre-aggregated by province, with the **mode** used for NPK levels and the **mean** for pH ranges.
-   **Feature Extraction:**
    -   The `app.services.feature_extractor.FeatureExtractor` service is used to generate **9 core features** for the model. These features are numerical scores (typically 0-100) that quantify the match between farmer input and crop requirements.
    -   **Feature List:**
        1.  `npk_match`: Score based on how well farmer's NPK levels match the crop's needs.
        2.  `ph_proximity`: Score based on how close the farmer's pH is to the crop's ideal range.
        3.  `temp_suitability`: Score for temperature match.
        4.  `rainfall_suitability`: Score for rainfall match.
        5.  `humidity_suitability`: Score for humidity match.
        6.  `soil_match`: Score based on soil type compatibility.
        7.  `historical_yield`: Normalized score of past yields for that crop in the province.
        8.  `season_alignment`: Score indicating if the current month is a good planting season for the crop.
        9.  `regional_success`: A broader measure of the crop's success in that geographical region.
-   **Target Variable Creation:**
    -   The model does not predict yield directly. Instead, it predicts a `suitability_score`.
    -   This target variable is created by normalizing the actual `yield_per_ha` for a crop against its **95th percentile yield**.
    -   **Formula:** `suitability_score = min(100.0, (actual_yield / 95th_percentile_yield) * 80 + 20)`
    -   This transforms yield into a stable, bounded metric of performance, making it a more robust target for the regression model.
-   **Output:** The process generates `training_dataset.csv` and `validation_dataset.csv`, which are saved to the `/models` directory.

---

## 5. Phase 4: Model Training

The model is trained in the `notebooks/03_model_training.ipynb` notebook.

-   **Objective:** To train and evaluate multiple regression models to find the best one for predicting the `suitability_score`.
-   **Models Trained:**
    -   `RandomForestRegressor` (from scikit-learn)
    -   `XGBRegressor` (from XGBoost)
    -   `GradientBoostingRegressor` (from scikit-learn)
-   **Hyperparameter Tuning:** `GridSearchCV` is used to perform a 3-fold cross-validation to find the optimal hyperparameters for each model, optimizing for `neg_mean_squared_error`.
-   **Best Model Selection:** The models are compared on the validation set using Root Mean Squared Error (RMSE). The **Random Forest Regressor** consistently performs the best.
-   **Output:**
    -   The trained best-performing model is serialized using `joblib` and saved as `models/crop_suitability_model.pkl`.
    -   A `models/model_info.json` file is also created, storing the list of feature names and the type of the winning model (`RandomForest`).

---

## 6. Phase 5: Model Evaluation

The model's performance is formally evaluated in `notebooks/04_model_evaluation.ipynb`.

-   **Metrics:** The primary metrics used to evaluate the regression model are:
    -   **R-squared (R²):** Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
    -   **Mean Absolute Error (MAE):** Provides the average absolute difference between the predicted and actual scores.
    -   **Root Mean Squared Error (RMSE):** The standard deviation of the residuals (prediction errors).
-   **Interpretability:**
    -   **Feature Importance:** A plot is generated (`notebooks/feature_importance.png`) to show which of the 9 features most significantly influence the model's predictions.
    -   **Prediction Analysis:** A scatter plot of `predicted_vs_actual.png` is created to visually inspect the model's accuracy and identify any systematic biases.

---

## 7. Phase 6: Deployment and Serving

The trained model is deployed as part of the FastAPI application.

-   **API Framework:** **FastAPI** is used for its high performance and automatic interactive documentation.
-   **Serving:** The application is run using a Uvicorn ASGI server via the command: `uvicorn app.main:app --reload`.
-   **API Endpoints:**
    -   `POST /recommend`: The main endpoint for getting recommendations.
    -   `GET /health`: A health check endpoint that confirms the API is running and the model is loaded.
-   **Request/Response Schemas:**
    -   The `app.models.schemas` module defines strict Pydantic models for API contracts.
    -   `RecommendationRequest`: The input from the user, containing location, NPK levels, pH, and soil type.
    -   `RecommendationResponse`: The rich output, containing a ranked list of 10 `CropRecommendation` objects, each with detailed scores, breakdowns, and contextual information like expected yield, risks, and planting seasons.
-   **Prediction Logic:**
    1.  When a request hits `/recommend`, the `app.services.recommendation_engine.RecommendationEngine` is invoked.
    2.  For each potential crop, it calculates both a rule-based score and an ML score.
    3.  The `app.services.ml_model` service loads the `crop_suitability_model.pkl` file and uses it to predict the `suitability_score`.
    4.  The final `hybrid_score` is computed (40% rule-based, 60% ML).
    5.  The results are ranked, and the top 10 are returned in the structured response.

---

## 8. Dependencies and Environment

-   **Dependencies:** All required Python packages are listed in `requirements.txt`. Key libraries include:
    -   `fastapi` & `uvicorn`
    -   `pandas` & `numpy`
    -   `scikit-learn`
    -   `xgboost`
    -   `joblib`
-   **Setup:** The environment can be provisioned by running `pip install -r requirements.txt`.