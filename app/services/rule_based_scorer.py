"""
Rule-based scoring system (40% weight in hybrid score).
Implements expert knowledge-based scoring for crop suitability.
"""
from typing import Dict
import pandas as pd
from app.utils.data_processor import (
    parse_ph_range, 
    parse_soil_types,
    normalize_npk_level
)


class RuleBasedScorer:
    """Calculate rule-based scores for crop suitability."""
    
    def __init__(self):
        """Initialize rule-based scorer."""
        pass
    
    def calculate_score(
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
        avg_humidity: float
    ) -> Dict:
        """
        Calculate rule-based score (0-100) with component breakdown.
        
        Args:
            crop_data: Crop data from unified database
            farmer_nitrogen: Farmer's nitrogen level (Low/Medium/High)
            farmer_phosphorus: Farmer's phosphorus level (Low/Medium/High)
            farmer_potassium: Farmer's potassium level (Low/Medium/High)
            farmer_ph_min: Farmer's minimum pH
            farmer_ph_max: Farmer's maximum pH
            farmer_soil_type: Farmer's soil type
            avg_temperature: Average temperature (°C)
            avg_rainfall: Average rainfall (mm)
            avg_humidity: Average humidity (%)
        
        Returns:
            Dict with 'total_score' and 'breakdown' components
        """
        # Component 1: NPK Matching (30 points)
        npk_score = self._score_npk_matching(
            crop_data,
            farmer_nitrogen,
            farmer_phosphorus,
            farmer_potassium
        )
        
        # Component 2: pH Range Validation (20 points)
        ph_score = self._score_ph_range(
            crop_data,
            farmer_ph_min,
            farmer_ph_max
        )
        
        # Component 3: Temperature Suitability (20 points)
        temp_score = self._score_temperature(
            crop_data,
            avg_temperature
        )
        
        # Component 4: Rainfall Suitability (15 points)
        rainfall_score = self._score_rainfall(
            crop_data,
            avg_rainfall
        )
        
        # Component 5: Humidity Suitability (10 points)
        humidity_score = self._score_humidity(
            crop_data,
            avg_humidity
        )
        
        # Component 6: Soil Type Matching (5 points)
        soil_type_score = self._score_soil_type(
            crop_data,
            farmer_soil_type
        )
        
        # Total score
        total_score = (
            npk_score + ph_score + temp_score + 
            rainfall_score + humidity_score + soil_type_score
        )
        
        return {
            'total_score': total_score,
            'breakdown': {
                'npk': npk_score,
                'ph': ph_score,
                'temperature': temp_score,
                'rainfall': rainfall_score,
                'humidity': humidity_score,
                'soil_type': soil_type_score
            }
        }
    
    def _score_npk_matching(
        self,
        crop_data: pd.Series,
        farmer_nitrogen: str,
        farmer_phosphorus: str,
        farmer_potassium: str
    ) -> float:
        """
        Component 1: NPK Matching (30 points).
        10 points per nutrient if exact match, 0 otherwise.
        """
        score = 0.0
        
        # Normalize levels
        farmer_n = normalize_npk_level(farmer_nitrogen)
        farmer_p = normalize_npk_level(farmer_phosphorus)
        farmer_k = normalize_npk_level(farmer_potassium)
        
        # Get crop requirements
        crop_n = normalize_npk_level(crop_data.get('n_requirement', 'Medium'))
        crop_p = normalize_npk_level(crop_data.get('p_requirement', 'Medium'))
        crop_k = normalize_npk_level(crop_data.get('k_requirement', 'Medium'))
        
        # Check matches
        if farmer_n == crop_n:
            score += 10.0
        if farmer_p == crop_p:
            score += 10.0
        if farmer_k == crop_k:
            score += 10.0
        
        return score
    
    def _score_ph_range(
        self,
        crop_data: pd.Series,
        farmer_ph_min: float,
        farmer_ph_max: float
    ) -> float:
        """
        Component 2: pH Range Validation (20 points).
        Full score if farmer pH midpoint in crop range, otherwise penalize by distance.
        """
        # Parse crop pH range
        crop_ph_str = crop_data.get('Soil pH Range', '')
        crop_ph_min, crop_ph_max = parse_ph_range(crop_ph_str)
        
        # Calculate farmer pH midpoint
        farmer_ph_mid = (farmer_ph_min + farmer_ph_max) / 2
        
        # Check if midpoint is in crop range
        if crop_ph_min <= farmer_ph_mid <= crop_ph_max:
            return 20.0
        
        # Calculate distance from range
        if farmer_ph_mid < crop_ph_min:
            distance = crop_ph_min - farmer_ph_mid
        else:
            distance = farmer_ph_mid - crop_ph_max
        
        # Penalize: score = max(0, 20 - (distance × 10))
        score = max(0.0, 20.0 - (distance * 10))
        return score
    
    def _score_temperature(
        self,
        crop_data: pd.Series,
        avg_temperature: float
    ) -> float:
        """
        Component 3: Temperature Suitability (20 points).
        Full score if temp in range, otherwise penalize by distance.
        """
        crop_temp_min = crop_data.get('Temp_Min_C', 0)
        crop_temp_max = crop_data.get('Temp_Max_C', 50)
        
        if pd.isna(crop_temp_min) or pd.isna(crop_temp_max):
            return 10.0  # Neutral score if data missing
        
        # Check if temperature is in range
        if crop_temp_min <= avg_temperature <= crop_temp_max:
            return 20.0
        
        # Calculate distance
        if avg_temperature < crop_temp_min:
            distance = crop_temp_min - avg_temperature
        else:
            distance = avg_temperature - crop_temp_max
        
        # Penalize: score = max(0, 20 - (distance × 2))
        score = max(0.0, 20.0 - (distance * 2))
        return score
    
    def _score_rainfall(
        self,
        crop_data: pd.Series,
        avg_rainfall: float
    ) -> float:
        """
        Component 4: Rainfall Suitability (15 points).
        Full score if rainfall in range, otherwise penalize by distance.
        """
        crop_rain_min = crop_data.get('Rainfall_Min_mm', 0)
        crop_rain_max = crop_data.get('Rainfall_Max_mm', 5000)
        
        if pd.isna(crop_rain_min) or pd.isna(crop_rain_max):
            return 7.5  # Neutral score if data missing
        
        # Check if rainfall is in range
        if crop_rain_min <= avg_rainfall <= crop_rain_max:
            return 15.0
        
        # Calculate distance (normalized by 100mm)
        if avg_rainfall < crop_rain_min:
            distance = (crop_rain_min - avg_rainfall) / 100
        else:
            distance = (avg_rainfall - crop_rain_max) / 100
        
        # Penalize: score = max(0, 15 - (distance × 5))
        score = max(0.0, 15.0 - (distance * 5))
        return score
    
    def _score_humidity(
        self,
        crop_data: pd.Series,
        avg_humidity: float
    ) -> float:
        """
        Component 5: Humidity Suitability (10 points).
        Full score if humidity in range, otherwise penalize by distance.
        """
        crop_hum_min = crop_data.get('Humidity_Min_Percent', 0)
        crop_hum_max = crop_data.get('Humidity_Max_Percent', 100)
        
        if pd.isna(crop_hum_min) or pd.isna(crop_hum_max):
            return 5.0  # Neutral score if data missing
        
        # Check if humidity is in range
        if crop_hum_min <= avg_humidity <= crop_hum_max:
            return 10.0
        
        # Calculate distance to nearest boundary
        distance_to_min = abs(avg_humidity - crop_hum_min)
        distance_to_max = abs(avg_humidity - crop_hum_max)
        distance = min(distance_to_min, distance_to_max)
        
        # Penalize: score = max(0, 10 - distance)
        score = max(0.0, 10.0 - distance)
        return score
    
    def _score_soil_type(
        self,
        crop_data: pd.Series,
        farmer_soil_type: str
    ) -> float:
        """
        Component 6: Soil Type Matching (5 points).
        Full score if match, 1 point partial credit otherwise.
        """
        crop_soil_str = crop_data.get('Soil Type', '')
        acceptable_types = parse_soil_types(crop_soil_str)
        
        if not acceptable_types:
            return 1.0  # Partial credit if no data
        
        # Check if farmer soil type is in acceptable types (case-insensitive)
        farmer_soil_lower = farmer_soil_type.strip().lower()
        for soil_type in acceptable_types:
            if soil_type.strip().lower() == farmer_soil_lower:
                return 5.0
        
        return 1.0  # Partial credit for non-ideal soil

