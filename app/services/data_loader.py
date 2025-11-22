"""
Data loading and aggregation service.
Loads all CSV files and creates unified crop database with climate data.
"""
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class DataLoader:
    """Loads and aggregates all crop and climate datasets."""
    
    def __init__(self, data_dir: str = None):
        """Initialize data loader with data directory path."""
        if data_dir is None:
            # Try to find project root (look for raw_datasets directory)
            current = Path(__file__).resolve()
            # Go up from app/services/data_loader.py to project root
            project_root = current.parent.parent.parent
            self.data_dir = project_root / "raw_datasets"
        else:
            self.data_dir = Path(data_dir)
        self.crop_climate_req = None
        self.crop_requirements = None
        self.crop_npk_req = None
        self.historical_performance = None
        self.climate_data = None
        self.unified_crop_db = None
        self.climate_averages = {}  # Cache for climate averages
        self.historical_yield_cache = {}  # Cache for historical yield queries
        
    def load_all_data(self):
        """Load all CSV files into memory."""
        print("Loading datasets...")
        
        # Load crop climate requirements
        self.crop_climate_req = pd.read_csv(
            self.data_dir / "crop_climate_requirements.csv"
        )
        print(f"Loaded {len(self.crop_climate_req)} crop climate requirements")
        
        # Load crop requirements (pH, soil type, planting period, etc.)
        self.crop_requirements = pd.read_csv(
            self.data_dir / "crop_requirements.csv"
        )
        print(f"Loaded {len(self.crop_requirements)} crop requirements")
        
        # Load NPK requirements
        self.crop_npk_req = pd.read_csv(
            self.data_dir / "crop_npk_level_requirements.csv"
        )
        print(f"Loaded {len(self.crop_npk_req)} NPK requirements")
        
        # Load historical performance
        self.historical_performance = pd.read_csv(
            self.data_dir / "historical_crop_performance.csv"
        )
        print(f"Loaded {len(self.historical_performance)} historical performance records")
        
        # Load climate data (this is large, so we'll process it on demand)
        print("Loading climate data (this may take a moment)...")
        self.climate_data = pd.read_csv(
            self.data_dir / "climate_data.csv"
        )
        print(f"Loaded {len(self.climate_data)} climate records")
        
        # Create unified crop database
        self._create_unified_crop_db()
        
    def _create_unified_crop_db(self):
        """Merge all crop requirement tables into unified database."""
        print("Creating unified crop database...")
        
        # Start with crop requirements (has Crop_Name as 'Crop')
        unified = self.crop_requirements.copy()
        unified.rename(columns={'Crop': 'Crop_Name'}, inplace=True)
        
        # Merge climate requirements
        unified = unified.merge(
            self.crop_climate_req,
            on='Crop_Name',
            how='left'
        )
        
        # Merge NPK requirements (column name is 'crop')
        unified = unified.merge(
            self.crop_npk_req,
            left_on='Crop_Name',
            right_on='crop',
            how='left'
        )
        
        # Drop duplicate 'crop' column
        if 'crop' in unified.columns:
            unified.drop(columns=['crop'], inplace=True)
        
        self.unified_crop_db = unified
        print(f"Unified database created with {len(unified)} crops")
        
    def get_climate_averages(
        self, 
        province: str, 
        municipality: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get 5-year average climate data (2020-2024) for location.
        Falls back to province-level if municipality not found.
        
        Returns:
            Dict with 'temperature', 'rainfall', 'humidity'
        """
        # Normalize input
        province = province.strip().title()
        if municipality:
            municipality = municipality.strip().title()
        
        # Check cache
        cache_key = f"{province}_{municipality or 'province'}"
        if cache_key in self.climate_averages:
            return self.climate_averages[cache_key]
        
        # Filter for 2020-2024
        recent_data = self.climate_data[
            (self.climate_data['YEAR'] >= 2020) & 
            (self.climate_data['YEAR'] <= 2024)
        ].copy()
        
        # Try municipality first
        if municipality:
            location_data = recent_data[
                (recent_data['Province'].str.strip().str.title() == province) &
                (recent_data['Municipality'].str.strip().str.title() == municipality)
            ]
            
            if len(location_data) > 0:
                result = self._calculate_climate_averages(location_data)
                self.climate_averages[cache_key] = result
                return result
        
        # Fallback to province-level
        province_data = recent_data[
            recent_data['Province'].str.strip().str.title() == province
        ]
        
        if len(province_data) == 0:
            raise ValueError(
                f"No climate data found for province: {province}"
            )
        
        result = self._calculate_climate_averages(province_data)
        self.climate_averages[cache_key] = result
        return result
    
    def _calculate_climate_averages(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate averages for temperature, rainfall, and humidity."""
        # Filter by parameter type
        temp_data = data[data['PARAMETER'] == 'T2M']
        rainfall_data = data[data['PARAMETER'] == 'PRECTOTCORR_SUM']
        humidity_data = data[data['PARAMETER'] == 'RH2M']
        
        # Calculate averages from ANNUAL column
        avg_temp = temp_data['ANNUAL'].mean() if len(temp_data) > 0 else None
        avg_rainfall = rainfall_data['ANNUAL'].mean() if len(rainfall_data) > 0 else None
        avg_humidity = humidity_data['ANNUAL'].mean() if len(humidity_data) > 0 else None
        
        return {
            'temperature': avg_temp,
            'rainfall': avg_rainfall,
            'humidity': avg_humidity
        }
    
    def get_historical_yield(
        self, 
        crop_name: str, 
        province: str
    ) -> Optional[Dict[str, float]]:
        """
        Get historical yield data for crop in province.
        
        Returns:
            Dict with 'avg_yield_per_ha', 'years_of_data', 'total_records'
            or None if no data found
        """
        # Normalize names
        crop_name = crop_name.strip()
        province = province.strip().title()
        
        # Check cache first
        cache_key = f"{crop_name}_{province}"
        if cache_key in self.historical_yield_cache:
            return self.historical_yield_cache[cache_key]
        
        # Filter historical data
        crop_data = self.historical_performance[
            (self.historical_performance['Crop'].str.strip() == crop_name) &
            (self.historical_performance['Province'].str.strip().str.title() == province)
        ].copy()
        
        if len(crop_data) == 0:
            self.historical_yield_cache[cache_key] = None
            return None
        
        # Calculate yield per hectare
        crop_data['yield_per_ha'] = (
            crop_data['Volume_Production'] / crop_data['Area_Planted_Harvested']
        )
        
        # Remove any invalid yields (infinite or NaN)
        crop_data = crop_data[
            (crop_data['yield_per_ha'].notna()) & 
            (crop_data['yield_per_ha'] != float('inf'))
        ]
        
        if len(crop_data) == 0:
            self.historical_yield_cache[cache_key] = None
            return None
        
        avg_yield = crop_data['yield_per_ha'].mean()
        years_of_data = crop_data['Year'].nunique()
        total_records = len(crop_data)
        
        result = {
            'avg_yield_per_ha': avg_yield,
            'years_of_data': years_of_data,
            'total_records': total_records
        }
        
        # Cache the result
        self.historical_yield_cache[cache_key] = result
        return result
    
    def get_crop_by_name(self, crop_name: str) -> Optional[pd.Series]:
        """Get crop data by name from unified database."""
        if self.unified_crop_db is None:
            raise ValueError("Unified crop database not loaded. Call load_all_data() first.")
        
        crop = self.unified_crop_db[
            self.unified_crop_db['Crop_Name'].str.strip() == crop_name.strip()
        ]
        
        if len(crop) == 0:
            return None
        
        return crop.iloc[0]
    
    def get_all_crops(self) -> pd.DataFrame:
        """Get all crops from unified database."""
        if self.unified_crop_db is None:
            raise ValueError("Unified crop database not loaded. Call load_all_data() first.")
        
        return self.unified_crop_db.copy()

    

# Global instance (singleton pattern)
_data_loader_instance = None

def get_data_loader() -> DataLoader:
    """Get or create global data loader instance."""
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = DataLoader()
        _data_loader_instance.load_all_data()
    return _data_loader_instance

