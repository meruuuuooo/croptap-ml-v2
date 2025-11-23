# CropTap ML v3: Hybrid Crop Recommendation System

This project implements a hybrid crop recommendation system for Filipino farmers, combining rule-based logic with a machine learning model. It provides optimal crop recommendations via a FastAPI-based REST API, considering soil conditions, climate, and historical data.

The recommendation engine takes farmer inputs (province, municipality, soil NPK levels, pH, and soil type) and outputs a ranked list of the top 10 most suitable crops. Each recommendation includes a hybrid score (40% rule-based, 60% ML model), confidence, expected yield, and potential risks.

## Getting Started

Follow these steps to set up, train the model, and run the API server.

### 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

Before starting the API, the machine learning model must be trained. Navigate to the `notebooks/` directory and execute the following Jupyter notebooks in order:

1.  `01_data_exploration.ipynb`
2.  `02_feature_engineering.ipynb`
3.  `03_model_training.ipynb`
4.  `04_model_evaluation.ipynb`
5.  `05_model_deep_dive.ipynb` (Newly added for comprehensive model analysis)

This process will generate the `crop_suitability_model.pkl` file in the `models/` directory, which is essential for the API.

### 3. Run the API Server

Once the model is trained and the `crop_suitability_model.pkl` file exists, you can start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Project Structure

The project follows a modular and service-oriented architecture:

*   **`app/`**: Main application directory.
    *   **`main.py`**: FastAPI application entry point.
    *   **`models/`**: Pydantic schemas for API request/response validation.
    *   **`services/`**: Business logic for the recommendation engine, data loading, scoring, and prediction.
    *   **`utils/`**: Utility functions for data processing and validation.
*   **`notebooks/`**: Jupyter notebooks for data exploration, feature engineering, model training, and evaluation.
*   **`raw_datasets/`**: Raw CSV data for training and recommendations.
*   **`models/`**: Trained machine learning models and related information.
