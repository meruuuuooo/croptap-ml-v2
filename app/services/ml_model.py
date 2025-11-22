"""
Trained ML model service for predicting crop suitability scores.
Loads and uses trained regression model (Option 2: Direct suitability score prediction).
"""
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from app.services.feature_extractor import FeatureExtractor


class MLModelService:
    """Service for loading and using trained ML model."""
    
    def __init__(self, model_path: str = "models/crop_suitability_model.pkl", model_info_path: str = "models/model_info.json"):
        """Initialize ML model service."""
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.is_loaded = False
        self.feature_names = []
        self._load_model_info()
    
    def _load_model_info(self):
        """Load model info from JSON file."""
        if self.model_info_path.exists():
            with open(self.model_info_path, "r") as f:
                model_info = json.load(f)
                self.feature_names = model_info.get('feature_names', [])
        else:
            print(f"Warning: Model info file not found at {self.model_info_path}")

    def load_model(self) -> bool:
        """
        Load trained model from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            print(f"Warning: Model file not found at {self.model_path}")
            print("Please train the model first using the Jupyter notebooks.")
            return False
        
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"ML model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_score(
        self,
        crop_data,
        farmer_nitrogen: str,
        farmer_phosphorus: str,
        farmer_potassium: str,
        farmer_ph_min: float,
        farmer_ph_max: float,
        farmer_soil_type: str,
        avg_temperature: float,
        avg_rainfall: float,
        avg_humidity: float,
        historical_yield_data: Dict = None,
        current_month: int = None,
        province: str = None,
        crop_category: str = None,
        features: Dict = None
    ) -> float:
        """
        Predict suitability score (0-100) using trained ML model.
        
        Args:
            crop_data: Crop data from unified database
            farmer_nitrogen: Farmer's nitrogen level
            farmer_phosphorus: Farmer's phosphorus level
            farmer_potassium: Farmer's potassium level
            farmer_ph_min: Farmer's minimum pH
            farmer_ph_max: Farmer's maximum pH
            farmer_soil_type: Farmer's soil type
            avg_temperature: Average temperature
            avg_rainfall: Average rainfall
            avg_humidity: Average humidity
            historical_yield_data: Optional historical yield data
            current_month: Optional current month
            province: Optional province name (for encoding)
            crop_category: Optional crop category (for encoding)
            features: Optional pre-computed features dict (for performance)
        
        Returns:
            Predicted suitability score (0-100), or 50.0 if model not loaded
        """
        if not self.is_loaded:
            if not self.load_model():
                # Return neutral score if model not available
                return 50.0
        
        # Use pre-computed features if provided, otherwise extract them
        if features is None:
            features = self.feature_extractor.extract_features(
                crop_data=crop_data,
                farmer_nitrogen=farmer_nitrogen,
                farmer_phosphorus=farmer_phosphorus,
                farmer_potassium=farmer_potassium,
                farmer_ph_min=farmer_ph_min,
                farmer_ph_max=farmer_ph_max,
                farmer_soil_type=farmer_soil_type,
                avg_temperature=avg_temperature,
                avg_rainfall=avg_rainfall,
                avg_humidity=avg_humidity,
                historical_yield_data=historical_yield_data,
                current_month=current_month
            )
        
        # Prepare feature DataFrame (order and names must match training)
        if not self.feature_names:
            # This is a fallback, but the feature names should be loaded from model_info.json
            print("Warning: Feature names not loaded. Using hardcoded feature names.")
            self.feature_names = [
                'npk_match', 'ph_proximity', 'temp_suitability',
                'rainfall_suitability', 'humidity_suitability', 'soil_match',
                'historical_yield', 'season_alignment', 'regional_success'
            ]

        # Create a pandas DataFrame with the correct feature names
        feature_df = pd.DataFrame([features], columns=self.feature_names)
        
        try:
            # Predict suitability score
            # Passing a DataFrame with feature names avoids the UserWarning
            prediction = self.model.predict(feature_df)[0]
            
            # Ensure score is in 0-100 range
            prediction = max(0.0, min(100.0, prediction))
            
            return float(prediction)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 50.0  # Return neutral score on error


# Global instance
_ml_model_instance = None

def get_ml_model_service() -> MLModelService:
    """Get or create global ML model service instance."""
    global _ml_model_instance
    if _ml_model_instance is None:
        _ml_model_instance = MLModelService()
        _ml_model_instance.load_model()
    return _ml_model_instance

