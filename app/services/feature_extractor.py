"""
Feature extraction for ML model.
Extracts 9 normalized features (0-1 scale) from crop and farmer data.
"""
import pandas as pd
from datetime import datetime
import random
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
    
    def extract_features_vectorized(
        self,
        data: pd.DataFrame,
        crop_db: pd.DataFrame,
        current_month: int = None
    ) -> pd.DataFrame:
        """
        Extracts all 9 features in a vectorized manner.
        
        Args:
            data: Merged DataFrame with crop, province, soil, and climate data.
            crop_db: The unified crop database.
            current_month: Optional current month (1-12), defaults to now.

        Returns:
            DataFrame with the 9 extracted features.
        """
        if current_month is None:
            current_month = datetime.now().month

        # Merge with crop database to get crop-specific requirements
        df = pd.merge(data, crop_db, left_on='Crop', right_on='Crop_Name', how='left')
        
        features = pd.DataFrame(index=df.index)
        
        # Vectorized feature extraction
        features['npk_match'] = self._extract_npk_match_ratio(df)
        features['ph_proximity'] = self._extract_ph_proximity(df)
        features['temp_suitability'] = self._extract_temp_suitability(df)
        features['rainfall_suitability'] = self._extract_rainfall_suitability(df)
        features['humidity_suitability'] = self._extract_humidity_suitability(df)
        features['soil_match'] = self._extract_soil_match(df)
        features['historical_yield'] = self._extract_historical_yield(df)
        features['season_alignment'] = self._extract_season_alignment(df, current_month)
        features['regional_success'] = self._extract_regional_success(df)
        
        # Add back identifiers
        features['crop_name'] = df['Crop']
        features['province'] = df['Province']
        features['actual_yield'] = df['yield_per_ha']
        features['category'] = df['Category']
        
        return features
        
    def _extract_npk_match_ratio(self, df: pd.DataFrame) -> pd.Series:
        farmer_n = df['nitrogen'].apply(normalize_npk_level)
        farmer_p = df['phosphorus'].apply(normalize_npk_level)
        farmer_k = df['potassium'].apply(normalize_npk_level)
        
        crop_n = df['n_requirement'].fillna('Medium').apply(normalize_npk_level)
        crop_p = df['p_requirement'].fillna('Medium').apply(normalize_npk_level)
        crop_k = df['k_requirement'].fillna('Medium').apply(normalize_npk_level)
        
        matches = (farmer_n == crop_n).astype(int) + \
                  (farmer_p == crop_p).astype(int) + \
                  (farmer_k == crop_k).astype(int)

        return matches / 3.0

    def _extract_ph_proximity(self, df: pd.DataFrame) -> pd.Series:
        ph_ranges = df['Soil pH Range'].fillna('0-0').apply(parse_ph_range)
        crop_ph_min = ph_ranges.apply(lambda x: x[0])
        crop_ph_max = ph_ranges.apply(lambda x: x[1])
        
        crop_ph_optimal = (crop_ph_min + crop_ph_max) / 2
        crop_ph_range = crop_ph_max - crop_ph_min
        farmer_ph_mid = (df['ph_min'] + df['ph_max']) / 2
        
        ph_distance = (farmer_ph_mid - crop_ph_optimal).abs()
        ph_proximity = 1.0 - (ph_distance / crop_ph_range.replace(0, 1)) # Avoid division by zero
        return ph_proximity.clip(0, 1)

    def _extract_temp_suitability(self, df: pd.DataFrame) -> pd.Series:
        crop_temp_min = df['Temp_Min_C'].fillna(0)
        crop_temp_max = df['Temp_Max_C'].fillna(50)
        temp_optimal = (crop_temp_min + crop_temp_max) / 2
        temp_range = crop_temp_max - crop_temp_min
        
        temp_distance = (df['temperature'] - temp_optimal).abs()
        temp_suitability = 1.0 - (temp_distance / temp_range.replace(0, 1))
        return temp_suitability.clip(0, 1)

    def _extract_rainfall_suitability(self, df: pd.DataFrame) -> pd.Series:
        crop_rain_min = df['Rainfall_Min_mm'].fillna(0)
        crop_rain_max = df['Rainfall_Max_mm'].fillna(5000)
        rain_optimal = (crop_rain_min + crop_rain_max) / 2
        rain_range = crop_rain_max - crop_rain_min

        rain_distance = (df['rainfall'] - rain_optimal).abs()
        rainfall_suitability = 1.0 - (rain_distance / rain_range.replace(0, 1))
        return rainfall_suitability.clip(0, 1)

    def _extract_humidity_suitability(self, df: pd.DataFrame) -> pd.Series:
        crop_hum_min = df['Humidity_Min_Percent'].fillna(0)
        crop_hum_max = df['Humidity_Max_Percent'].fillna(100)
        hum_optimal = (crop_hum_min + crop_hum_max) / 2
        hum_range = crop_hum_max - crop_hum_min
        
        hum_distance = (df['humidity'] - hum_optimal).abs()
        humidity_suitability = 1.0 - (hum_distance / hum_range.replace(0, 1))
        return humidity_suitability.clip(0, 1)

    def _extract_soil_match(self, df: pd.DataFrame) -> pd.Series:
        df['acceptable_soils'] = df['Soil Type'].fillna('').apply(parse_soil_types)
        df['farmer_soil_type'] = df['acceptable_soils'].apply(lambda x: random.choice(x) if x else 'Loam')
        
        return df.apply(lambda row: 1.0 if row['farmer_soil_type'] in row['acceptable_soils'] else 0.3, axis=1)

    def _extract_historical_yield(self, df: pd.DataFrame) -> pd.Series:
        # This remains tricky to vectorize without a pre-computed lookup table.
        # For now, we'll keep it simple and normalize by a global max.
        avg_yield = df['yield_per_ha'].fillna(0)
        return (avg_yield / 20.0).clip(0, 1).fillna(0.5)

    def _extract_season_alignment(self, df: pd.DataFrame, current_month: int) -> pd.Series:
        planting_periods = df['Planting Period'].fillna('').apply(parse_planting_period)
        
        def is_in_season(period):
            if period is None: return 1.0
            return 1.0 if is_in_planting_season(period, current_month) else \
                   max(0.0, 1.0 - (calculate_distance_to_season(current_month, *period) / 6.0))

        return planting_periods.apply(is_in_season)

    def _extract_regional_success(self, df: pd.DataFrame) -> pd.Series:
        # Simplified for vectorization, based on number of records for that crop-province
        # A more accurate version would pre-aggregate years_of_data.
        return (df.groupby(['Crop', 'Province'])['Year'].transform('nunique') / 10.0).clip(0, 1).fillna(0.3)
