# Hybrid Crop Recommendation System - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Model Training](#model-training)
7. [API Documentation](#api-documentation)
8. [File Structure](#file-structure)
9. [Usage Instructions](#usage-instructions)
10. [Technical Specifications](#technical-specifications)

---

## System Overview

The Hybrid Crop Recommendation System is an intelligent decision support tool designed for Filipino farmers. It combines rule-based expert knowledge with machine learning to recommend optimal crops based on:

- **Soil Conditions**: NPK levels (Nitrogen, Phosphorus, Potassium), pH range, and soil type
- **Climate Data**: Temperature, rainfall, and humidity averages (2020-2024)
- **Historical Performance**: Past crop yields in the farmer's province
- **Crop Requirements**: 106 crops with comprehensive requirement specifications

### Key Features

- **Hybrid Scoring Approach**: Combines two scoring methods:
  - Rule-based scoring (50% weight) - Expert knowledge with 6 core components
  - Trained ML model scoring (50% weight) - Data-driven predictions using 9 features
- **Comprehensive Analysis**: Evaluates all 106 crops simultaneously
- **Risk Assessment**: Identifies potential issues and provides recommendations
- **Yield Prediction**: Estimates expected yields based on conditions
- **RESTful API**: Easy integration with web/mobile applications

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                      (app/main.py)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│  Data Loader │ │  Validator  │ │   Engine   │
└───────┬──────┘ └─────────────┘ └─────┬──────┘
        │                               │
        │         ┌─────────────────────┼─────────────────────┐
        │         │                                           │
┌───────▼─────────▼──────┐                             ┌──────▼──────────┐
│  Rule-Based Scorer     │                             │  ML Model       │
│  (50% weight)          │                             │  (50% weight)   │
│                        │                             │                 │
└────────────────────────┘                             └─────────────────┘
```

### Data Flow

1. **Input**: Farmer provides location, soil conditions (NPK, pH, soil type)
2. **Data Loading**: System loads and aggregates all datasets
3. **Climate Retrieval**: Fetches 5-year climate averages for location
4. **Crop Evaluation**: For each of 106 crops:
   - Calculate rule-based score
   - Extract 9 features and calculate feature-based score
   - Predict ML model score
   - Combine scores (hybrid = 40% rule + 60% ML)
5. **Ranking**: Sort crops by hybrid score
6. **Output**: Return top 10 recommendations with detailed analysis

---

## Data Sources

### 1. Crop Climate Requirements (`crop_climate_requirements.csv`)
- **104 crops** with temperature, rainfall, and humidity requirements
- Columns: `Crop_Name`, `Temp_Min_C`, `Temp_Max_C`, `Rainfall_Min_mm`, `Rainfall_Max_mm`, `Humidity_Min_Percent`, `Humidity_Max_Percent`

### 2. Crop Requirements (`crop_requirements.csv`)
- Soil pH ranges, soil types, planting periods, days to harvest, yield ranges
- Columns: `Crop`, `Category`, `Soil pH Range`, `Soil Type`, `Planting Period`, `Days to Harvest`, `Yield Per Hectare`

### 3. NPK Level Requirements (`crop_npk_level_requirements.csv`)
- Nutrient requirements for each crop
- Columns: `crop`, `category`, `n_requirement`, `p_requirement`, `k_requirement`, `nutrient_notes`

### 4. Climate Data (`climate_data.csv`)
- **215,556 records** of historical climate data (1981-2024)
- Parameters: `T2M` (Temperature), `PRECTOTCORR_SUM` (Rainfall), `RH2M` (Humidity)
- Granularity: Province, Municipality, Year, Monthly + Annual averages

### 5. Historical Crop Performance (`historical_crop_performance.csv`)
- **93,369 records** of actual crop yields (2010-2024)
- Columns: `Crop`, `Province`, `Year`, `Volume_Production`, `Area_Planted_Harvested`
- Used to calculate yield per hectare and regional success rates

### 6. Soil Test Data (`soil_test_data.csv`)
- **22,242 records** of actual farmer soil test results
- Columns: `province`, `municipality`, `farmer`, `nitrogen`, `phosphorus`, `potassium`, `ph_min`, `ph_max`
- Used for model training and province-level soil condition aggregation

---

## Methodology

### Hybrid Scoring System

The system uses a two-tier scoring approach:

#### 1. Rule-Based Scoring (50% Weight)

Implements expert knowledge through 6 components (total 100 points):

**Component 1: NPK Matching (30 points)**
- Exact match = 10 points per nutrient (N, P, K)
- Formula: `score = (N_match ? 10 : 0) + (P_match ? 10 : 0) + (K_match ? 10 : 0)`

**Component 2: pH Range Validation (20 points)**
- Full score if farmer pH midpoint falls within crop's pH range
- Otherwise: `score = max(0, 20 - (distance × 10))`

**Component 3: Temperature Suitability (20 points)**
- Full score if average temperature within crop range
- Otherwise: `score = max(0, 20 - (distance × 2))`

**Component 4: Rainfall Suitability (15 points)**
- Full score if average rainfall within crop range
- Otherwise: `score = max(0, 15 - (distance/100 × 5))`

**Component 5: Humidity Suitability (10 points)**
- Full score if average humidity within crop range
- Otherwise: `score = max(0, 10 - distance)`

**Component 6: Soil Type Matching (5 points)**
- Full score if farmer's soil type in crop's acceptable types
- Partial credit (1 point) otherwise

**Total Rule Score**: Sum of all components (0-100)

#### 2. Trained ML Model Scoring (50% Weight)

**Model Type**: Regression (Random Forest or XGBoost)

**Target Variable**: Suitability Score (0-100)
- Derived from historical yields: `score = min(100, (yield/20) × 80 + 20)`
- High yield + consistent performance → 80-100
- Medium yield → 50-80
- Low yield → 0-50

**Features**: 9 normalized features (0-1 scale):
1. **NPK Match Ratio**: Count of matching nutrients / 3
2. **pH Proximity**: Distance from optimal pH within range
3. **Temperature Suitability**: Distance from optimal temp within range
4. **Rainfall Suitability**: Distance from optimal rainfall within range
5. **Humidity Suitability**: Distance from optimal humidity within range
6. **Soil Type Match**: Binary/partial match (1.0 or 0.3)
7. **Historical Yield Performance**: Normalized yield per hectare (max 20 tons/ha)
8. **Growing Season Alignment**: Current month in planting period
9. **Regional Success Rate**: Years of data / 10 (max 1.0)

**Key Advantage**: The ML model learns complex patterns and interactions between all 9 features, including the 3 unique features (historical yield, season alignment, regional success) that are not in rule-based scoring.

**Training Data**: 
- Merged dataset combining:
  - Historical crop performance (actual yields)
  - Soil test data (real farmer conditions)
  - Climate data (2020-2024 averages)
- **4,884 training records** with real data

**Model Output**: Direct suitability score prediction (0-100)

#### Final Hybrid Score

```
HYBRID_SCORE = (RULE_SCORE × 0.50) + (ML_MODEL_SCORE × 0.50)
```

**Rationale**: 
- Rule-based (50%): Provides interpretable expert knowledge based on 6 core factors
- ML Model (50%): Learns data-driven patterns from all 9 features, including historical/seasonal/regional context

#### Confidence Calculation

```
CONFIDENCE = 100 - |RULE_SCORE - ML_MODEL_SCORE|
```

**Confidence Levels**:
- High confidence (≥80): Rule-based and ML model agree closely
- Medium confidence (60-79): Some disagreement between methods
- Low confidence (<60): Significant disagreement - may indicate edge case or data quality issue

---

## Implementation Details

### Data Loading Service (`app/services/data_loader.py`)

**Responsibilities**:
- Load all 6 CSV files into memory
- Create unified crop database by merging requirement tables
- Calculate 5-year climate averages (2020-2024)
- Handle missing data with province-level fallback
- Cache climate averages for performance
- Aggregate soil test data by province for training

**Key Methods**:
- `load_all_data()`: Loads all datasets and creates unified database
- `get_climate_averages(province, municipality)`: Returns temperature, rainfall, humidity averages
- `get_historical_yield(crop_name, province)`: Returns yield statistics for crop-province
- `get_crop_by_name(crop_name)`: Retrieves complete crop data

**Climate Data Processing**:
```python
# Filter for 2020-2024
recent_data = climate_data[(climate_data['YEAR'] >= 2020) & (climate_data['YEAR'] <= 2024)]

# Calculate averages by parameter type
temp_avg = recent_data[recent_data['PARAMETER'] == 'T2M']['ANNUAL'].mean()
rainfall_avg = recent_data[recent_data['PARAMETER'] == 'PRECTOTCORR_SUM']['ANNUAL'].mean()
humidity_avg = recent_data[recent_data['PARAMETER'] == 'RH2M']['ANNUAL'].mean()
```

### Rule-Based Scorer (`app/services/rule_based_scorer.py`)

Implements all 6 scoring components with detailed breakdown:

**NPK Matching**:
```python
def _score_npk_matching(crop_data, farmer_n, farmer_p, farmer_k):
    score = 0.0
    if farmer_n == crop_data['n_requirement']: score += 10.0
    if farmer_p == crop_data['p_requirement']: score += 10.0
    if farmer_k == crop_data['k_requirement']: score += 10.0
    return score
```

**pH Range Validation**:
```python
def _score_ph_range(crop_data, farmer_ph_min, farmer_ph_max):
    crop_ph_min, crop_ph_max = parse_ph_range(crop_data['Soil pH Range'])
    farmer_ph_mid = (farmer_ph_min + farmer_ph_max) / 2
    
    if crop_ph_min <= farmer_ph_mid <= crop_ph_max:
        return 20.0
    
    distance = min(abs(farmer_ph_mid - crop_ph_min), 
                   abs(farmer_ph_mid - crop_ph_max))
    return max(0.0, 20.0 - (distance * 10))
```

Similar implementations for temperature, rainfall, humidity, and soil type.

### Feature Extractor (`app/services/feature_extractor.py`)

Extracts 9 normalized features (0-1 scale) for ML model input and yield prediction:

**Example: Temperature Suitability**:
```python
def _extract_temp_suitability(crop_data, avg_temperature):
    crop_temp_min = crop_data['Temp_Min_C']
    crop_temp_max = crop_data['Temp_Max_C']
    temp_optimal = (crop_temp_min + crop_temp_max) / 2
    temp_range = crop_temp_max - crop_temp_min
    temp_distance = abs(avg_temperature - temp_optimal)
    
    return max(0.0, 1.0 - (temp_distance / temp_range))
```

**Season Alignment**:
```python
def _extract_season_alignment(crop_data, current_month):
    planting_period = parse_planting_period(crop_data['Planting Period'])
    if planting_period is None:  # All season
        return 1.0
    
    if is_in_planting_season(planting_period, current_month):
        return 1.0
    
    months_away = calculate_distance_to_season(...)
    return max(0.0, 1.0 - (months_away / 6.0))
```

### ML Scorer (Removed)

The feature-based scoring component has been removed to eliminate redundancy. The ML model now directly uses all 9 features, including the 3 unique features (historical yield, season alignment, regional success) that provide additional context beyond rule-based scoring.

### ML Model Service (`app/services/ml_model.py`)

**Model Loading**:
- Loads trained model from `models/crop_yield_model.pkl`
- Handles missing model gracefully (returns neutral score 50.0)

**Prediction Process**:
1. Extract 9 features using FeatureExtractor
2. Prepare feature array in correct order
3. Predict suitability score (0-100)
4. Ensure score is within valid range

### Recommendation Engine (`app/services/recommendation_engine.py`)

**Main Orchestrator**:

```python
def generate_recommendations(province, municipality, nitrogen, phosphorus, 
                            potassium, ph_min, ph_max, soil_type, top_n=10):
    # 1. Get climate data
    climate = data_loader.get_climate_averages(province, municipality)
    
    # 2. Loop through all crops
    for crop in all_crops:
        # 3. Calculate rule-based score (6 components)
        rule_score = rule_scorer.calculate_score(...)
        
        # 4. Predict ML model score (uses all 9 features)
        ml_model_score = ml_model.predict_score(...)
        
        # 5. Calculate hybrid score (50/50 split)
        hybrid_score = (rule_score * 0.50 + ml_model_score * 0.50)
        
        # 6. Calculate confidence (agreement between two methods)
        confidence = 100 - |rule_score - ml_model_score|
        
        # 7. Extract features for yield prediction
        features = feature_extractor.extract_features(...)
        
        # 8. Identify risks
        risks = identify_risks(rule_breakdown, historical_data)
        
        # 9. Predict yield
        expected_yield = yield_predictor.predict_yield(...)
    
    # 10. Sort and return top N
    recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return recommendations[:top_n]
```

**Risk Assessment**:
- NPK mismatch if rule NPK score < 20
- pH out of range if rule pH score < 15
- Temperature issues if rule temp score < 15
- Rainfall issues if rule rainfall score < 10
- Limited history if historical_yield < 0.3

### Yield Predictor (`app/services/yield_predictor.py`)

**Prediction Logic**:
```python
if historical_data exists:
    base_yield = historical_data['avg_yield_per_ha']
    climate_factor = (temp_suitability + rainfall_suitability + humidity_suitability) / 3
    soil_factor = (npk_match + ph_proximity + soil_match) / 3
    expected_yield = base_yield × climate_factor × soil_factor
else:
    expected_yield = parse_yield_range(crop_data['Yield Per Hectare'])
```

### Utility Functions (`app/utils/`)

**Data Processor** (`data_processor.py`):
- `parse_ph_range()`: Parses "6.0-7.0" → (6.0, 7.0)
- `parse_soil_types()`: Parses "Loam, Clay loam" → ["Loam", "Clay loam"]
- `parse_yield_range()`: Parses "2-4 tons" → (2.0, 4.0)
- `parse_planting_period()`: Parses "May-Aug" → (5, 8)
- `is_in_planting_season()`: Checks if current month in planting period
- `normalize_npk_level()`: Normalizes to "Low", "Medium", or "High"

**Validators** (`validators.py`):
- Input validation for NPK levels, pH ranges, soil types
- Location validation against climate data
- Comprehensive farmer input validation

---

## Model Training

### Training Dataset Creation

**Process** (see `notebooks/02_feature_engineering.ipynb`):

1. **Load Historical Performance Data**
   - Calculate yield per hectare: `yield = Volume_Production / Area_Planted_Harvested`
   - Filter valid yields (> 0, not infinite)

2. **Load and Aggregate Soil Test Data**
   - Group by province
   - Get most common NPK levels (mode)
   - Calculate average pH ranges

3. **Merge Datasets**
   ```python
   merged_data = historical_perf.merge(
       province_soil_agg,
       on='Province_normalized',
       how='inner'
   )
   ```

4. **Extract Features for Each Record**
   - Use real soil conditions from merged data
   - Use real climate averages
   - Use actual yields as targets

5. **Create Target Variable**
   ```python
   suitability_score = min(100.0, (yield_per_ha / 20.0) * 80.0 + 20.0)
   ```

**Result**: 4,884 training records with real data

### Model Training Process

**Notebook**: `notebooks/03_model_training.ipynb`

1. **Load Training Dataset**
   ```python
   train_df = pd.read_csv("../models/training_dataset.csv")
   X = train_df[feature_columns]  # 9 features
   y = train_df['suitability_score']  # Target (0-100)
   ```

2. **Split Data**
   - 80% training, 20% testing
   - Random state = 42 for reproducibility

3. **Train Random Forest**
   ```python
   rf_model = RandomForestRegressor(
       n_estimators=100,
       max_depth=10,
       min_samples_split=5,
       min_samples_leaf=2,
       random_state=42
   )
   ```

4. **Train XGBoost**
   ```python
   xgb_model = xgb.XGBRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       random_state=42
   )
   ```

5. **Select Best Model**
   - Compare RMSE (Root Mean Squared Error)
   - Save best model to `models/crop_yield_model.pkl`

6. **Evaluation Metrics**
   - RMSE: Root Mean Squared Error
   - MAE: Mean Absolute Error
   - R²: Coefficient of Determination

### Model Evaluation

**Notebook**: `notebooks/04_model_evaluation.ipynb`

- Feature importance analysis
- Prediction vs actual scatter plots
- Residual analysis
- Performance metrics on full dataset

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

#### 2. Health Check
```
GET /health
```
Returns system health status and loaded data statistics.

**Response**:
```json
{
  "status": "healthy",
  "crops_loaded": 104,
  "ml_model_loaded": true
}
```

#### 3. Get Recommendations
```
POST /recommend
```

**Request Body**:
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

**Response**:
```json
{
  "location": {
    "province": "Cagayan",
    "municipality": "Tuguegarao"
  },
  "climate_summary": {
    "avg_temperature": 27.5,
    "avg_rainfall": 1850.0,
    "avg_humidity": 78.0
  },
  "soil_summary": {
    "nitrogen": "Medium",
    "phosphorus": "High",
    "potassium": "Medium",
    "ph_range": "6.0-6.8",
    "soil_type": "Loam"
  },
  "recommendations": [
    {
      "rank": 1,
      "crop_name": "Mais (Corn)",
      "category": "Corn",
      "hybrid_score": 87.5,
      "rule_score": 85.0,
      "ml_model_score": 88.1,
      "confidence": 97.1,
      "badge": "Best Match",
      "expected_yield": "4.5-6.0 tons/ha",
      "risks": [],
      "planting_season": "May-Aug",
      "days_to_harvest": "90-120 DAP",
      "fertilizer_recommendation": "Apply 120-150 kg N/ha split application",
      "why_recommended": "Excellent NPK match, ideal climate conditions, proven success in Cagayan with avg 5.2 tons/ha over past 5 years",
      "rule_breakdown": {
        "npk": 30.0,
        "ph": 20.0,
        "temperature": 20.0,
        "rainfall": 15.0,
        "humidity": 10.0,
        "soil_type": 5.0
      }
    }
    // ... 9 more recommendations
  ],
  "insights": {
    "best_season": "May-August for optimal planting",
    "soil_amendments": "Soil conditions are suitable for most crops",
    "diversification_tip": "Consider mixing Mais (Corn) with Sitaw (Cowpea) for crop rotation"
  }
}
```

**Error Responses**:

400 Bad Request - Validation Error:
```json
{
  "detail": "Invalid NPK level: Invalid. Must be one of ['Low', 'Medium', 'High']"
}
```

404 Not Found - Location Not Found:
```json
{
  "detail": "No climate data found for province: InvalidProvince"
}
```

500 Internal Server Error:
```json
{
  "detail": "Internal server error: [error message]"
}
```

---

## File Structure

```
croptap-ml-v2/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Data loading and aggregation
│   │   ├── rule_based_scorer.py     # Rule-based scoring (40%)
│   │   ├── feature_extractor.py     # Feature extraction for ML
│   │   ├── ml_scorer.py             # Feature-based scoring (30%)
│   │   ├── ml_model.py              # Trained ML model service (30%)
│   │   ├── recommendation_engine.py # Main recommendation orchestrator
│   │   └── yield_predictor.py       # Yield prediction
│   └── utils/
│       ├── __init__.py
│       ├── data_processor.py        # Data parsing utilities
│       └── validators.py            # Input validation
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data exploration and analysis
│   ├── 02_feature_engineering.ipynb # Training dataset creation
│   ├── 03_model_training.ipynb      # Model training
│   └── 04_model_evaluation.ipynb    # Model evaluation
├── models/
│   ├── crop_yield_model.pkl         # Trained ML model (generated)
│   ├── training_dataset.csv         # Training dataset (generated)
│   └── model_info.json              # Model metadata (generated)
├── raw_datasets/
│   ├── climate_data.csv
│   ├── crop_climate_requirements.csv
│   ├── crop_npk_level_requirements.csv
│   ├── crop_requirements.csv
│   ├── historical_crop_performance.csv
│   └── soil_test_data.csv
├── requirements.txt
├── README.md
├── documentation.md                 # This file
└── .gitignore
```

---

## Usage Instructions

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd croptap-ml-v2
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Model Training

1. **Run Jupyter Notebooks in Order**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_feature_engineering.ipynb
   jupyter notebook notebooks/03_model_training.ipynb
   jupyter notebook notebooks/04_model_evaluation.ipynb
   ```

2. **Verify Model Creation**:
   ```bash
   ls models/crop_yield_model.pkl
   ```

### Running the API

1. **Start the Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access API Documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "province": "Cagayan",
       "municipality": "Tuguegarao",
       "nitrogen": "Medium",
       "phosphorus": "High",
       "potassium": "Medium",
       "ph_min": 6.0,
       "ph_max": 6.8,
       "soil_type": "Loam"
     }'
   ```

### Using Python Client

```python
import requests

url = "http://localhost:8000/recommend"
payload = {
    "province": "Cagayan",
    "municipality": "Tuguegarao",
    "nitrogen": "Medium",
    "phosphorus": "High",
    "potassium": "Medium",
    "ph_min": 6.0,
    "ph_max": 6.8,
    "soil_type": "Loam"
}

response = requests.post(url, json=payload)
recommendations = response.json()

for rec in recommendations['recommendations']:
    print(f"{rec['rank']}. {rec['crop_name']} - Score: {rec['hybrid_score']}")
```

---

## Technical Specifications

### Performance Metrics

- **Response Time**: < 2 seconds for 106 crops evaluation
- **Accuracy**: Model trained on 4,884 real data records
- **Coverage**: 104 crops, 80 provinces, 62 provinces with soil data

### Data Processing

- **Climate Data**: 215,556 records processed
- **Historical Performance**: 93,369 yield records
- **Soil Test Data**: 22,242 farmer records aggregated to 62 provinces
- **Training Dataset**: 4,884 records with complete feature set

### Algorithm Complexity

- **Time Complexity**: O(n × m) where:
  - n = number of crops (106)
  - m = feature extraction + model prediction complexity (constant)
- **Space Complexity**: O(n) for storing crop data and recommendations
- **Scoring Components**: 2 (rule-based + ML model) instead of 3, reducing computation

### Scalability

- **Horizontal Scaling**: Stateless API design allows multiple instances
- **Caching**: Climate averages cached to reduce computation
- **Database Ready**: Can be migrated to database for larger datasets

### Limitations

1. **Soil Type**: Currently uses default "Loam" (can be improved with soil type mapping)
2. **Model Training**: Uses aggregated province-level soil data (not individual farmer data)
3. **Climate Fallback**: Falls back to province-level if municipality not found
4. **Season Alignment**: Uses current month (could be parameterized)
5. **Feature-Based Score Removed**: The 3 unique features (historical yield, season, regional success) are now only in ML model, not in a separate interpretable score

### Future Improvements

1. **Real-time Climate Data**: Integration with weather APIs
2. **Individual Farmer Profiles**: Store and learn from farmer-specific data
3. **Market Price Integration**: Include market demand and pricing
4. **Multi-crop Rotation**: Recommend crop sequences
5. **Mobile App**: Native mobile application for farmers
6. **Offline Mode**: Local model for areas with limited connectivity

---

## References

### Technologies Used

- **Python 3.10+**: Programming language
- **FastAPI**: Web framework for API
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Pydantic**: Data validation
- **Jupyter**: Interactive notebooks for analysis

### Data Sources

- Philippine Statistics Authority (PSA) - Historical crop performance
- PAGASA - Climate data
- Department of Agriculture - Crop requirements and soil data

---

## Conclusion

The Hybrid Crop Recommendation System successfully combines rule-based expert knowledge with machine learning to provide accurate, actionable crop recommendations for Filipino farmers. The system processes real-world data from multiple sources, evaluates 106 crops simultaneously, and provides detailed insights to help farmers make informed decisions.

The two-tier scoring approach (rule-based 50% + ML model 50%) ensures robust recommendations with confidence metrics. The rule-based component provides interpretable expert knowledge, while the ML model learns complex patterns from historical data including seasonal and regional factors. This design eliminates redundancy while maintaining both interpretability and data-driven insights. The comprehensive risk assessment helps farmers understand potential challenges and required interventions.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: Crop Recommendation System Development Team

