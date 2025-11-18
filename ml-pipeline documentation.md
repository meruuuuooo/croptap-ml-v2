# CropTap ML Pipeline Documentation

This document provides a comprehensive, top-to-bottom overview of the machine learning pipeline for the CropTap recommendation system. It details each phase, from data ingestion to model deployment, reflecting the project's specific implementation and best practices.

## 1. Project Overview

The CropTap system is a **hybrid crop recommendation engine** designed for Filipino farmers. Its primary goal is to provide optimal crop recommendations by combining two distinct methodologies to leverage their respective strengths:

1.  **Rule-Based System:** A static, expert-driven scoring system that matches farmer inputs (e.g., soil NPK, pH, climate conditions) against predefined optimal ranges and conditions for each crop. This system captures domain-specific agricultural knowledge.
2.  **Machine Learning Model:** A data-driven predictive model trained on historical performance and environmental data. It forecasts a "suitability score" based on complex patterns identified in past data, offering adaptability and insight beyond explicit rules.

The final recommendation is derived from a **hybrid score**, calculated as a weighted average. Specifically, the rule-based system contributes **40%** to the final score, while the ML model contributes **60%**. This weighting scheme was determined through experimentation to provide a balanced and accurate recommendation. The entire system is exposed as a robust REST API built with **FastAPI**, enabling seamless integration with various front-end applications.

## 2. Phase 1: Data Ingestion

This phase is responsible for efficiently loading all necessary raw data from persistent storage into the application's memory for processing.

-   **Sources:** Data is sourced from a collection of well-structured static CSV files located in the `raw_datasets/` directory. Each file represents a specific aspect of the agricultural ecosystem.
-   **Process:** The `app.services.data_loader.DataLoader` class implements a **singleton pattern** to ensure that data is loaded only once upon application startup. This design choice optimizes memory usage and reduces I/O operations, ensuring high performance. The `DataLoader` parses each CSV file into a `pandas.DataFrame` and stores it in an in-memory dictionary for quick access. Robust error handling is implemented to manage scenarios such as missing files, corrupted CSVs, or parsing errors, logging issues and preventing application crashes.
-   **Key Datasets:**
    -   `historical_crop_performance.csv`: Contains records of historical crop yields (volume production and area planted/harvested) across different geographical locations (provinces, municipalities) and years. Key columns include `Province`, `Municipality`, `Crop`, `Year`, `Area Planted`, `Area Harvested`, `Volume Production`, `Yield per Hectare`.
    -   `soil_test_data.csv`: Provides aggregated real-world soil analysis information, including Nitrogen (N), Phosphorus (P), Potassium (K) levels, and pH values. This data is typically aggregated by `Province`, ensuring general soil characteristics for larger regions. Key columns include `Province`, `N`, `P`, `K`, `pH_min`, `pH_max`, `Soil Type`.
    -   `climate_data.csv`: Contains 5-year average climate data for various municipalities, including temperature (`Avg_Temp_C`), rainfall (`Avg_Rainfall_mm`), and humidity (`Avg_Humidity_Percent`). Key columns include `Province`, `Municipality`, `Month`, `Avg_Temp_C`, `Avg_Rainfall_mm`, `Avg_Humidity_Percent`.
    -   `crop_requirements.csv`, `crop_npk_level_requirements.csv`, `crop_climate_requirements.csv`: These files collectively store the rule-based information detailing ideal growth conditions for each crop. They define optimal ranges for NPK, pH, temperature, rainfall, humidity, and preferred soil types. Key columns across these files include `Crop`, `Min_N`, `Max_N`, `Min_P`, `Max_P`, `Min_K`, `Max_K`, `Min_pH`, `Max_pH`, `Ideal_Soil_Type`, `Min_Temp_C`, `Max_Temp_C`, etc.

## 3. Phase 2: Data Exploration

An exploratory data analysis (EDA) is performed through the `notebooks/01_data_exploration.ipynb` notebook to gain a deep understanding of the raw data's characteristics, identify potential issues, and inform subsequent feature engineering steps.

-   **Objective:** To systematically analyze data distributions, detect missing values, identify outliers, uncover correlations and relationships between variables, and understand historical yield patterns and geographical variations. The process utilizes `pandas` functions like `df.info()`, `df.describe()`, `df.isnull().sum()`, and `df['column'].value_counts()` for initial data assessment.
-   **Key Findings:**
    -   **Crop Yield Distribution:** Crop yields were observed to be heavily right-skewed, with a large number of lower-yield records and a smaller number of exceptionally high-yield outliers. This skewness suggests varying agricultural practices, environmental conditions, or data reporting inconsistencies across different regions and years.
    -   **Top-Performing Crops:** Initial analysis identified crops like Cassava and Sweet Potato as consistently exhibiting high average yields, providing insights into potentially robust crops for the region.
    -   **Geographical Data Density:** The density and completeness of data varied significantly across provinces, indicating potential limitations for highly localized recommendations in certain areas.
    -   **Missing Values:** Assessment of missing values guided imputation or removal strategies for feature engineering.
-   **Visualizations:** The notebook generates a comprehensive suite of plots using `matplotlib` and `seaborn` for visual inspection and documentation. These include:
    -   Histograms of crop yields to visualize their distribution.
    -   Bar charts illustrating top 20 crops and provinces by average yield.
    -   Line charts showing average crop yield trends over time.
    -   Heatmaps depicting correlation matrices for production volume, area, and yield.
    -   Histograms of climate variables (temperature, rainfall, humidity) to understand their ranges and distributions.
    -   Geospatial plots (if data permits) showing yield distribution by province.

## 4. Phase 3: Feature Engineering

This phase, meticulously documented and executed in `notebooks/02_feature_engineering.ipynb`, is critical for transforming raw data into a clean, normalized, and feature-rich dataset suitable for training the machine learning model.

-   **Data Merging:**
    1.  The `historical_crop_performance.csv` serves as the base for building the training dataset.
    2.  It is then merged with `soil_test_data.csv` using the `Province` column as the key, typically employing a `left` merge to retain all historical crop performance records.
    3.  The `soil_test_data` is pre-aggregated by province, with the **mode** used for categorical NPK levels (if applicable) and soil type, and the **mean** used for numerical pH ranges, ensuring a representative soil profile for each province.
    4.  Similarly, `climate_data.csv` is merged to incorporate average climate conditions, often using `Province` and `Month` as merge keys.
-   **Feature Extraction (9 Core Features):**
    The `app.services.feature_extractor.FeatureExtractor` service is central to generating the **9 core features** that quantify the suitability of a crop for given farmer inputs. Each feature is designed to be a numerical score, typically normalized within a range (e.g., 0-100), to quantify the "match" between farmer input conditions and the crop's requirements.

    1.  **`npk_match`:** Calculated by assessing how closely the farmer's current soil NPK levels (`N`, `P`, `K`) align with the ideal NPK ranges (`Min_N`, `Max_N`, etc.) for a specific crop. This can be scored inversely proportional to the deviation from the ideal range, or using a binning approach where scores are higher for being within the optimal range.
    2.  **`ph_proximity`:** Scores the closeness of the farmer's soil pH to the crop's ideal pH range (`Min_pH`, `Max_pH`). A score of 100 might indicate being within the ideal range, with scores decreasing as the pH deviates further.
    3.  **`temp_suitability`:** Evaluates how well the average regional temperature (`Avg_Temp_C`) aligns with the crop's optimal temperature requirements (`Min_Temp_C`, `Max_Temp_C`).
    4.  **`rainfall_suitability`:** Quantifies the match between regional average rainfall (`Avg_Rainfall_mm`) and the crop's water requirements.
    5.  **`humidity_suitability`:** Scores the compatibility of regional average humidity (`Avg_Humidity_Percent`) with the crop's preferred humidity levels.
    6.  **`soil_match`:** A categorical or numerical score indicating the compatibility of the farmer's soil type (`Soil Type`) with the `Ideal_Soil_Type` for the crop. This might involve one-hot encoding or a similarity metric.
    7.  **`historical_yield`:** A normalized score reflecting the past performance of that specific crop in the farmer's `Province` or `Municipality`. This could be calculated as `(crop_yield_in_region / max_crop_yield_overall) * 100` to provide a relative measure.
    8.  **`season_alignment`:** A score indicating whether the current month or planting period aligns with the optimal planting season for the crop. This could be a binary (0/1) or graded score based on proximity to ideal months.
    9.  **`regional_success`:** A broader measure of the crop's overall success or prevalence in the broader geographical region (e.g., province or district), considering aggregate historical production volumes.
-   **Target Variable Creation (`suitability_score`):**
    -   The model is designed to predict a `suitability_score` rather than raw yield. This is a more stable and robust target variable, as raw yield can be highly volatile due to various external factors.
    -   The `suitability_score` is created by normalizing the actual `yield_per_ha` for a given crop against its **95th percentile yield**. The 95th percentile is used to make the target robust to extreme outliers in yield data, providing a more realistic upper bound for "excellent" performance.
    -   **Formula:** `suitability_score = min(100.0, (actual_yield_per_ha / 95th_percentile_yield_for_crop) * 80 + 20)`
        -   The `(actual_yield / 95th_percentile_yield)` component normalizes the yield relative to top performance.
        -   Multiplying by `80` and adding `20` scales the score to a range of `[20, 100]`, making it interpretable as a percentage-like suitability while avoiding scores of zero which might imply impossibility.
    -   This transformation ensures a bounded and more meaningful target for a regression model, representing the crop's inherent potential suitability under given conditions.
-   **Output:** The phase culminates in the generation of `training_dataset.csv` and `validation_dataset.csv` files, which are saved to the `/models` directory. The `sklearn.model_selection.train_test_split` function is used to split the prepared data, ensuring a robust evaluation of the trained model on unseen data.

## 5. Phase 4: Model Training

The training of the machine learning model is meticulously conducted and documented in the `notebooks/03_model_training.ipynb` notebook.

-   **Objective:** To rigorously train and evaluate several regression models to identify the most effective one for predicting the `suitability_score`. The objective is to achieve high predictive accuracy and generalization capabilities.
-   **Models Trained:**
    -   `RandomForestRegressor` (from `scikit-learn`): Chosen for its robustness, ability to handle non-linear relationships, and feature importance insights.
    -   `XGBRegressor` (from `XGBoost` library): Selected for its superior performance in structured data, speed, and advanced regularization techniques.
    -   `GradientBoostingRegressor` (from `scikit-learn`): Included as a powerful ensemble method and a good baseline for comparison with XGBoost.
-   **Hyperparameter Tuning:** `sklearn.model_selection.GridSearchCV` is extensively used to systematically search for the optimal hyperparameters for each model. This involves:
    -   Defining a comprehensive parameter grid for each model (e.g., for `RandomForestRegressor`: `n_estimators` [100, 200, 300], `max_depth` [10, 20, None], `min_samples_split` [2, 5, 10]).
    -   Performing a **3-fold cross-validation** to ensure the robustness of the selected hyperparameters and prevent overfitting to a single train-validation split.
    -   Optimizing for `neg_mean_squared_error` as the scoring metric, which `GridSearchCV` aims to maximize (equivalent to minimizing Mean Squared Error).
-   **Best Model Selection:** After cross-validation and hyperparameter tuning, models are compared based on their performance on the validation set, primarily using **Root Mean Squared Error (RMSE)**. The **Random Forest Regressor** consistently demonstrated the best balance of predictive accuracy and generalization performance during this comparative analysis, making it the chosen model for deployment.
-   **Output:**
    -   The trained best-performing model (Random Forest Regressor) is serialized using `joblib.dump` and saved to `models/crop_suitability_model.pkl`. This allows for efficient loading and deployment without retraining.
    -   A `models/model_info.json` file is also created, storing crucial metadata such as the list of feature names used during training and the type of the selected model (`RandomForest`), ensuring consistency during prediction.

## 6. Phase 5: Model Evaluation

The formal evaluation of the chosen model's performance is conducted within the `notebooks/04_model_evaluation.ipynb` notebook, using established regression metrics and interpretability techniques.

-   **Metrics:** The primary metrics employed to evaluate the regression model's accuracy and reliability are:
    -   **R-squared (R²):** Measures the proportion of the variance in the dependent variable (`suitability_score`) that is predictable from the independent variables (features). An R² value closer to 1 indicates a model that explains a larger proportion of the variance in the target variable, signifying a better fit.
    -   **Mean Absolute Error (MAE):** Provides the average absolute difference between the predicted `suitability_score` and the actual `suitability_score`. MAE is robust to outliers and offers a straightforward interpretation of the average prediction error in the same units as the target variable. Lower MAE values indicate higher accuracy.
    -   **Root Mean Squared Error (RMSE):** Represents the standard deviation of the residuals (prediction errors). RMSE gives a relatively high weight to large errors, making it particularly useful when large errors are undesirable. Like MAE, lower RMSE values indicate better model performance, and it is also in the same units as the target variable.
-   **Interpretability:**
    -   **Feature Importance:** The `model.feature_importances_` attribute (available for tree-based models like Random Forest) is used to generate a plot (`notebooks/feature_importance.png`). This plot visually represents which of the 9 features contributed most significantly to the model's predictions. This insight is crucial for understanding the model's decision-making process and can guide future feature engineering or data collection efforts.
    -   **Prediction Analysis:** A scatter plot of `predicted_vs_actual.png` is generated. This visualization plots the model's predicted `suitability_score` against the actual `suitability_score`. An ideal model's predictions would fall perfectly on the `y=x` line. Deviations from this line reveal systematic biases (e.g., under-prediction or over-prediction in certain ranges) or areas where the model struggles. Residual plots are also analyzed to check for homoscedasticity and detect patterns in errors.

## 7. Phase 6: Deployment and Serving

The culmination of the ML pipeline is the deployment of the trained model as a robust and scalable web service using FastAPI.

-   **API Framework:** **FastAPI** is chosen for its modern features, high performance (being built on Starlette and Pydantic), automatic interactive API documentation (Swagger UI / ReDoc), and intuitive request/response validation.
-   **Serving:** The application is served using `uvicorn`, an ASGI server, via the command: `uvicorn app.main:app --reload`. The `--reload` flag is particularly useful during development as it automatically reloads the server upon code changes.
-   **API Endpoints:**
    -   `POST /recommend`: This is the core endpoint for receiving farmer inputs and returning crop recommendations. It expects a JSON request body conforming to the `RecommendationRequest` schema.
    -   `GET /health`: A standard health check endpoint that confirms the API is running, the model is successfully loaded into memory, and essential data dependencies are accessible.
-   **Request/Response Schemas:**
    -   The `app.models.schemas` module meticulously defines strict Pydantic models, serving as clear contracts for API input and output.
    -   `RecommendationRequest`: Defines the expected input from the user, including `province`, `municipality`, `soil_n_level`, `soil_p_level`, `soil_k_level`, `soil_ph`, and `soil_type`. These inputs are validated automatically by FastAPI.
    -   `RecommendationResponse`: Defines the rich output structure. It contains a ranked list of up to 10 `CropRecommendation` objects. Each `CropRecommendation` object provides detailed insights:
        -   `crop_name`: The name of the recommended crop.
        -   `hybrid_score`: The final weighted suitability score.
        -   `rule_based_score`: The score derived solely from the rule-based system.
        -   `ml_model_score`: The suitability score predicted by the machine learning model.
        -   `confidence_level`: An estimation of the recommendation's reliability.
        -   `expected_yield`: A predicted yield range.
        -   `potential_risks`: Identified risks associated with growing the crop in the given conditions.
        -   `planting_seasons`: Optimal planting times.
        -   Other relevant contextual information.
-   **Prediction Logic:**
    1.  Upon receiving a request at `/recommend`, the `app.services.recommendation_engine.RecommendationEngine` orchestrates the entire prediction process.
    2.  For each potential crop in the system's catalog, the engine first invokes the `app.services.rule_based_scorer.RuleBasedScorer` to calculate a score based on predefined agricultural rules and the farmer's inputs.
    3.  Concurrently, the `app.services.ml_scorer.MLScorer` prepares the necessary features (the 9 core features) from the farmer's inputs and loaded data. It then calls the `app.services.ml_model` service, which has loaded the `crop_suitability_model.pkl` file, to predict the `ml_model_score` (the suitability score).
    4.  The `app.services.yield_predictor.YieldPredictor` also uses the model's output and historical data to estimate `expected_yield`.
    5.  The final `hybrid_score` is computed using the weighted average: `hybrid_score = (rule_based_score * 0.4) + (ml_model_score * 0.6)`.
    6.  The system then ranks all potential crops based on their `hybrid_score` in descending order.
    7.  The top 10 recommendations, enriched with all the detailed contextual information, are then returned in the structured `RecommendationResponse`.

## 8. Dependencies and Environment

-   **Dependencies:** All required Python packages are meticulously listed in `requirements.txt` to ensure a consistent and reproducible development and deployment environment. Key libraries include:
    -   `fastapi>=0.104.1`: The web framework for building the API.
    -   `uvicorn[standard]>=0.23.2`: The ASGI server for running the FastAPI application.
    -   `pandas>=2.0.3`: Essential for data manipulation, cleaning, and analysis.
    -   `numpy>=1.25.2`: Fundamental package for numerical computing.
    -   `scikit-learn>=1.3.0`: Provides various machine learning algorithms, including `RandomForestRegressor`, `GridSearchCV`, and utility functions.
    -   `xgboost>=1.7.6`: High-performance gradient boosting library.
    -   `joblib>=1.3.2`: Used for efficient serialization and deserialization of Python objects, particularly the trained ML model.
    -   `matplotlib>=3.7.2`: For creating static, interactive, and animated visualizations in Python.
    -   `seaborn>=0.12.2`: Statistical data visualization library based on matplotlib.
    -   `jupyter`: For interactive development and documentation of notebooks.
-   **Setup:** The development environment can be quickly provisioned by navigating to the project root and executing the following command: `pip install -r requirements.txt`. This command installs all specified dependencies into the active Python environment.
