"""
Feature extraction for ML model.
Extracts 9 normalized features (0-1 scale) from crop and farmer data.
"""
from typing import Dict
import pandas as pd
from datetime import datetime
from app.utils.data_processor import (
    parse_ph_range,
    parse_soil_types,
    parse_planting_period,
    is_in_planting_season,
    calculate_distance_to_season,
    normalize_npk_level
)


class FeatureExtractor:
    """Extract normalized features for ML model input."""
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(
        self,
        crop_data: pd.Series,
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
    ) -> Dict[str, float]:
        """
        Extract all 9 features (normalized 0-1 scale).
        
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
            historical_yield_data: Optional dict with historical yield info
            current_month: Optional current month (1-12), defaults to now
        
        Returns:
            Dict with 9 feature values (0-1 scale)
        """
        if current_month is None:
            current_month = datetime.now().month
        
        features = {}
        
        # Feature 1: NPK Match Ratio
        features['npk_match'] = self._extract_npk_match_ratio(
            crop_data, farmer_nitrogen, farmer_phosphorus, farmer_potassium
        )
        
        # Feature 2: pH Proximity
        features['ph_proximity'] = self._extract_ph_proximity(
            crop_data, farmer_ph_min, farmer_ph_max
        )
        
        # Feature 3: Temperature Suitability
        features['temp_suitability'] = self._extract_temp_suitability(
            crop_data, avg_temperature
        )
        
        # Feature 4: Rainfall Suitability
        features['rainfall_suitability'] = self._extract_rainfall_suitability(
            crop_data, avg_rainfall
        )
        
        # Feature 5: Humidity Suitability
        features['humidity_suitability'] = self._extract_humidity_suitability(
            crop_data, avg_humidity
        )
        
        # Feature 6: Soil Type Match
        features['soil_match'] = self._extract_soil_match(
            crop_data, farmer_soil_type
        )
        
        # Feature 7: Historical Yield Performance
        features['historical_yield'] = self._extract_historical_yield(
            historical_yield_data
        )
        
        # Feature 8: Growing Season Alignment
        features['season_alignment'] = self._extract_season_alignment(
            crop_data, current_month
        )
        
        # Feature 9: Regional Success Rate
        features['regional_success'] = self._extract_regional_success(
            historical_yield_data
        )
        
        return features
    
    def _extract_npk_match_ratio(
        self,
        crop_data: pd.Series,
        farmer_nitrogen: str,
        farmer_phosphorus: str,
        farmer_potassium: str
    ) -> float:
        """Feature 1: NPK Match Ratio (0, 0.33, 0.67, or 1.0)."""
        matches = 0
        
        farmer_n = normalize_npk_level(farmer_nitrogen)
        farmer_p = normalize_npk_level(farmer_phosphorus)
        farmer_k = normalize_npk_level(farmer_potassium)
        
        crop_n = normalize_npk_level(crop_data.get('n_requirement', 'Medium'))
        crop_p = normalize_npk_level(crop_data.get('p_requirement', 'Medium'))
        crop_k = normalize_npk_level(crop_data.get('k_requirement', 'Medium'))
        
        if farmer_n == crop_n:
            matches += 1
        if farmer_p == crop_p:
            matches += 1
        if farmer_k == crop_k:
            matches += 1
        
        return matches / 3.0
    
    def _extract_ph_proximity(
        self,
        crop_data: pd.Series,
        farmer_ph_min: float,
        farmer_ph_max: float
    ) -> float:
        """Feature 2: pH Proximity (0-1)."""
        crop_ph_str = crop_data.get('Soil pH Range', '')
        crop_ph_min, crop_ph_max = parse_ph_range(crop_ph_str)
        
        if crop_ph_max == crop_ph_min:
            return 1.0  # Perfect match if single value
        
        crop_ph_optimal = (crop_ph_min + crop_ph_max) / 2
        crop_ph_range = crop_ph_max - crop_ph_min
        farmer_ph_mid = (farmer_ph_min + farmer_ph_max) / 2
        
        ph_distance = abs(farmer_ph_mid - crop_ph_optimal)
        ph_proximity = max(0.0, 1.0 - (ph_distance / crop_ph_range))
        
        return ph_proximity
    
    def _extract_temp_suitability(
        self,
        crop_data: pd.Series,
        avg_temperature: float
    ) -> float:
        """Feature 3: Temperature Suitability (0-1)."""
        crop_temp_min = crop_data.get('Temp_Min_C', 0)
        crop_temp_max = crop_data.get('Temp_Max_C', 50)
        
        if pd.isna(crop_temp_min) or pd.isna(crop_temp_max):
            return 0.5  # Neutral if missing
        
        if crop_temp_max == crop_temp_min:
            return 1.0 if avg_temperature == crop_temp_min else 0.0
        
        temp_optimal = (crop_temp_min + crop_temp_max) / 2
        temp_range = crop_temp_max - crop_temp_min
        temp_distance = abs(avg_temperature - temp_optimal)
        
        temp_suitability = max(0.0, 1.0 - (temp_distance / temp_range))
        return temp_suitability
    
    def _extract_rainfall_suitability(
        self,
        crop_data: pd.Series,
        avg_rainfall: float
    ) -> float:
        """Feature 4: Rainfall Suitability (0-1)."""
        crop_rain_min = crop_data.get('Rainfall_Min_mm', 0)
        crop_rain_max = crop_data.get('Rainfall_Max_mm', 5000)
        
        if pd.isna(crop_rain_min) or pd.isna(crop_rain_max):
            return 0.5  # Neutral if missing
        
        if crop_rain_max == crop_rain_min:
            return 1.0 if avg_rainfall == crop_rain_min else 0.0
        
        rain_optimal = (crop_rain_min + crop_rain_max) / 2
        rain_range = crop_rain_max - crop_rain_min
        rain_distance = abs(avg_rainfall - rain_optimal)
        
        rainfall_suitability = max(0.0, 1.0 - (rain_distance / rain_range))
        return rainfall_suitability
    
    def _extract_humidity_suitability(
        self,
        crop_data: pd.Series,
        avg_humidity: float
    ) -> float:
        """Feature 5: Humidity Suitability (0-1)."""
        crop_hum_min = crop_data.get('Humidity_Min_Percent', 0)
        crop_hum_max = crop_data.get('Humidity_Max_Percent', 100)
        
        if pd.isna(crop_hum_min) or pd.isna(crop_hum_max):
            return 0.5  # Neutral if missing
        
        if crop_hum_max == crop_hum_min:
            return 1.0 if avg_humidity == crop_hum_min else 0.0
        
        hum_optimal = (crop_hum_min + crop_hum_max) / 2
        hum_range = crop_hum_max - crop_hum_min
        hum_distance = abs(avg_humidity - hum_optimal)
        
        humidity_suitability = max(0.0, 1.0 - (hum_distance / hum_range))
        return humidity_suitability
    
    def _extract_soil_match(
        self,
        crop_data: pd.Series,
        farmer_soil_type: str
    ) -> float:
        """Feature 6: Soil Type Match (0.3 or 1.0)."""
        crop_soil_str = crop_data.get('Soil Type', '')
        acceptable_types = parse_soil_types(crop_soil_str)
        
        if not acceptable_types:
            return 0.3  # Partial score if no data
        
        farmer_soil_lower = farmer_soil_type.strip().lower()
        for soil_type in acceptable_types:
            if soil_type.strip().lower() == farmer_soil_lower:
                return 1.0
        
        return 0.3  # Partial score for non-ideal soil
    
    def _extract_historical_yield(
        self,
        historical_yield_data: Dict = None
    ) -> float:
        """Feature 7: Historical Yield Performance (0-1)."""
        if historical_yield_data is None:
            return 0.5  # Neutral score for untested crops
        
        avg_yield = historical_yield_data.get('avg_yield_per_ha', 0)
        
        # Normalize: Assume max expected yield is 20 tons/ha
        historical_yield = min(1.0, avg_yield / 20.0)
        return historical_yield
    
    def _extract_season_alignment(
        self,
        crop_data: pd.Series,
        current_month: int
    ) -> float:
        """Feature 8: Growing Season Alignment (0-1)."""
        planting_period_str = crop_data.get('Planting Period', '')
        planting_period = parse_planting_period(planting_period_str)
        
        if planting_period is None:
            return 1.0  # All season
        
        if is_in_planting_season(planting_period, current_month):
            return 1.0
        
        # Calculate months until next planting window
        start_month, end_month = planting_period
        months_away = calculate_distance_to_season(
            current_month, start_month, end_month
        )
        
        # Decay over 6 months
        season_alignment = max(0.0, 1.0 - (months_away / 6.0))
        return season_alignment
    
    def _extract_regional_success(
        self,
        historical_yield_data: Dict = None
    ) -> float:
        """Feature 9: Regional Success Rate (0-1)."""
        if historical_yield_data is None:
            return 0.3  # Low confidence for crops never grown locally
        
        years_of_data = historical_yield_data.get('years_of_data', 0)
        
        # More years = more proven in the region (max 10 years)
        regional_success = min(1.0, years_of_data / 10.0)
        return regional_success

