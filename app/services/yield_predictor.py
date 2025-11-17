"""
Yield prediction service.
Predicts expected yield based on historical data and condition factors.
"""
from typing import Dict, Optional
import pandas as pd
from app.utils.data_processor import parse_yield_range


class YieldPredictor:
    """Predict expected crop yield."""
    
    def __init__(self):
        """Initialize yield predictor."""
        pass
    
    def predict_yield(
        self,
        crop_data: pd.Series,
        historical_yield_data: Optional[Dict] = None,
        climate_factor: float = 1.0,
        soil_factor: float = 1.0
    ) -> str:
        """
        Predict expected yield per hectare.
        
        Args:
            crop_data: Crop data from unified database
            historical_yield_data: Optional historical yield data for province
            climate_factor: Climate suitability factor (0-1)
            soil_factor: Soil suitability factor (0-1)
        
        Returns:
            Formatted yield string like "4.5-6.0 tons/ha"
        """
        # If historical data exists, use it as base
        if historical_yield_data and historical_yield_data.get('avg_yield_per_ha'):
            base_yield = historical_yield_data['avg_yield_per_ha']
            
            # Adjust based on conditions
            adjusted_yield = base_yield * climate_factor * soil_factor
            
            # Create range (±20% of adjusted yield)
            min_yield = adjusted_yield * 0.8
            max_yield = adjusted_yield * 1.2
            
            return f"{min_yield:.1f}-{max_yield:.1f} tons/ha"
        
        # Otherwise, use crop requirements yield range
        yield_str = crop_data.get('Yield Per Hectare', '')
        if pd.isna(yield_str) or not yield_str:
            return "Yield data not available"
        
        # Parse yield range from crop requirements
        min_yield, max_yield = parse_yield_range(yield_str)
        
        if min_yield == 0 and max_yield == 0:
            return "Yield data not available"
        
        # Adjust by factors
        min_yield_adj = min_yield * climate_factor * soil_factor
        max_yield_adj = max_yield * climate_factor * soil_factor
        
        return f"{min_yield_adj:.1f}-{max_yield_adj:.1f} tons/ha"
    
    def calculate_climate_factor(
        self,
        temp_suitability: float,
        rainfall_suitability: float,
        humidity_suitability: float
    ) -> float:
        """
        Calculate climate factor from suitability scores.
        
        Args:
            temp_suitability: Temperature suitability (0-1)
            rainfall_suitability: Rainfall suitability (0-1)
            humidity_suitability: Humidity suitability (0-1)
        
        Returns:
            Climate factor (0-1)
        """
        return (temp_suitability + rainfall_suitability + humidity_suitability) / 3.0
    
    def calculate_soil_factor(
        self,
        npk_match: float,
        ph_proximity: float,
        soil_match: float
    ) -> float:
        """
        Calculate soil factor from suitability scores.
        
        Args:
            npk_match: NPK match ratio (0-1)
            ph_proximity: pH proximity (0-1)
            soil_match: Soil type match (0-1)
        
        Returns:
            Soil factor (0-1)
        """
        return (npk_match + ph_proximity + soil_match) / 3.0

