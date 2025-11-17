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
            # Try to find project root, assuming script is in app/services
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "raw_datasets"
            # Fallback for notebook execution where __file__ might be misleading
            if not self.data_dir.exists():
                self.data_dir = Path.cwd() / "raw_datasets"
        else:
            self.data_dir = Path(data_dir)

        self.crop_climate_req = None
        self.crop_requirements = None
        self.crop_npk_req = None
        self.historical_performance = None
        self.climate_data = None
        self.unified_crop_db = None
        self.climate_averages = None  # Will be a DataFrame

    def load_all_data(self):
        """Load all CSV files into memory and pre-calculate averages."""
        print("Loading datasets...")

        self.crop_climate_req = pd.read_csv(self.data_dir / "crop_climate_requirements.csv")
        self.crop_requirements = pd.read_csv(self.data_dir / "crop_requirements.csv")
        self.crop_npk_req = pd.read_csv(self.data_dir / "crop_npk_level_requirements.csv")
        self.historical_performance = pd.read_csv(self.data_dir / "historical_crop_performance.csv")
        self.climate_data = pd.read_csv(self.data_dir / "climate_data.csv")

        print("Datasets loaded. Creating unified database and climate averages...")
        self._create_unified_crop_db()
        self._pre_calculate_climate_averages()
        print("Initialization complete.")

    def _create_unified_crop_db(self):
        """Merge all crop requirement tables into a unified database."""
        unified = self.crop_requirements.copy().rename(columns={'Crop': 'Crop_Name'})
        unified = pd.merge(unified, self.crop_climate_req, on='Crop_Name', how='left')
        unified = pd.merge(unified, self.crop_npk_req, left_on='Crop_Name', right_on='crop', how='left')
        if 'crop' in unified.columns:
            unified.drop(columns=['crop'], inplace=True)
        self.unified_crop_db = unified

    def _pre_calculate_climate_averages(self):
        """Pre-calculate and cache 5-year climate averages for all provinces."""
        recent_data = self.climate_data[self.climate_data['YEAR'].between(2020, 2024)].copy()
        recent_data['Province'] = recent_data['Province'].str.strip().str.title()

        # Pivot and calculate mean
        pivot = recent_data.pivot_table(
            index='Province', columns='PARAMETER', values='ANNUAL', aggfunc='mean'
        )
        self.climate_averages = pivot.rename(columns={
            'T2M': 'temperature',
            'PRECTOTCORR_SUM': 'rainfall',
            'RH2M': 'humidity'
        })
        print(f"Pre-calculated climate averages for {len(self.climate_averages)} provinces.")

    def get_climate_averages(self, province: str) -> Optional[Dict[str, float]]:
        """Retrieve pre-calculated climate averages for a province."""
        province = province.strip().title()
        if province in self.climate_averages.index:
            return self.climate_averages.loc[province].to_dict()
        return None

    def get_historical_yield(self, crop_name: str, province: str) -> Optional[Dict[str, float]]:
        """Get historical yield data for a crop in a province."""
        crop_name = crop_name.strip()
        province = province.strip().title()
        
        crop_data = self.historical_performance[
            (self.historical_performance['Crop'].str.strip() == crop_name) &
            (self.historical_performance['Province'].str.strip().str.title() == province)
        ].copy()

        if crop_data.empty:
            return None

        crop_data['yield_per_ha'] = crop_data['Volume_Production'] / crop_data['Area_Planted_Harvested']
        crop_data.dropna(subset=['yield_per_ha'], inplace=True)
        crop_data = crop_data[crop_data['yield_per_ha'] != float('inf')]

        if crop_data.empty:
            return None

        return {
            'avg_yield_per_ha': crop_data['yield_per_ha'].mean(),
            'years_of_data': crop_data['Year'].nunique(),
            'total_records': len(crop_data)
        }

    def get_crop_by_name(self, crop_name: str) -> Optional[pd.Series]:
        """Get crop data by name from the unified database."""
        crop = self.unified_crop_db[self.unified_crop_db['Crop_Name'].str.strip() == crop_name.strip()]
        return crop.iloc[0] if not crop.empty else None

    def get_all_crops(self) -> pd.DataFrame:
        """Get all crops from the unified database."""
        return self.unified_crop_db.copy()

# Singleton pattern
_data_loader_instance = None
def get_data_loader() -> DataLoader:
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = DataLoader()
        _data_loader_instance.load_all_data()
    return _data_loader_instance
