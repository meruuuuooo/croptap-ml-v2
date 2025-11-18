"""
Feature-based ML scorer (30% weight in hybrid score).
Calculates weighted combination of extracted features.
"""
from typing import Dict
from app.services.feature_extractor import FeatureExtractor


class MLScorer:
    """Calculate feature-based ML score using weighted feature combination."""
    
    # Feature weights (must sum to 1.0)
    FEATURE_WEIGHTS = {
        'historical_yield': 0.260,
        'ph_proximity': 0.205,
        'temp_suitability': 0.166,
        'humidity_suitability': 0.157,
        'rainfall_suitability': 0.068,
        'soil_match': 0.048,
        'npk_match': 0.040,
        'season_alignment': 0.034,
        'regional_success': 0.022
    }
    
    def __init__(self):
        """Initialize ML scorer."""
        self.feature_extractor = FeatureExtractor()
    
    def calculate_score(
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
        current_month: int = None
    ) -> Dict:
        """
        Calculate feature-based ML score (0-100) with feature breakdown.
        
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
        
        Returns:
            Dict with 'total_score' and 'breakdown' of features
        """
        # Extract features
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
        
        # Calculate weighted score
        total_score = 0.0
        for feature_name, feature_value in features.items():
            weight = self.FEATURE_WEIGHTS.get(feature_name, 0.0)
            total_score += feature_value * weight
        
        # Scale to 0-100
        total_score *= 100
        
        return {
            'total_score': total_score,
            'breakdown': features
        }

