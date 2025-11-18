# Project Overview

This project is a hybrid crop recommendation system designed for Filipino farmers. It combines a rule-based approach with a machine learning model to provide optimal crop recommendations based on soil conditions, climate, and historical performance data. The system is delivered as a REST API using FastAPI.

The core of the system is a recommendation engine that takes farmer inputs (province, municipality, soil NPK levels, pH, and soil type) and returns a ranked list of the top 10 most suitable crops. Each recommendation is enriched with a hybrid score, a rule-based score, an ML model score, a confidence level, expected yield, potential risks, and other useful information.

The hybrid scoring is weighted, with 40% of the score coming from the rule-based system and 60% from the machine learning model.

## Building and Running

### 1. Install Dependencies

To set up the project, install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

Before running the application, the machine learning model must be trained. This is done by executing the Jupyter notebooks in the `notebooks/` directory in the following order:

1.  `01_data_exploration.ipynb`
2.  `02_feature_engineering.ipynb`
3.  `03_model_training.ipynb`
4.  `04_model_evaluation.ipynb`

This process will generate the `crop_suitability_model.pkl` file in the `models/` directory.

### 3. Run the API Server

Once the model is trained, you can start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Development Conventions

The application is structured into several modules:

*   **`app/`**: The main application directory.
    *   **`main.py`**: The FastAPI application entry point.
    *   **`models/`**: Contains Pydantic schemas for API request and response validation.
    *   **`services/`**: Contains the business logic for the recommendation engine, data loading, scoring, and prediction.
    *   **`utils/`**: Contains utility functions for data processing and validation.
*   **`notebooks/`**: Jupyter notebooks for data exploration, feature engineering, model training, and evaluation.
*   **`raw_datasets/`**: Contains the raw CSV data used for training and recommendations.
*   **`models/`**: Contains the trained machine learning model and model information.

The code follows a modular and service-oriented architecture, with a clear separation of concerns between the API layer, business logic, and data access.
