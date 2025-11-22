import pandas as pd
import numpy as np
import os

# --- Province Name Standardization Mapping ---
# This mapping is derived from observed discrepancies in the data.
# It converts common variations and abbreviations to a consistent format (Title Case).
PROVINCE_MAPPING = {
    'ABRA': 'Abra',
    'AGUSAN DEL NORTE': 'Agusan del Norte',
    'AGUSAN DEL SUR': 'Agusan del Sur',
    'AKLAN': 'Aklan',
    'ALBAY': 'Albay',
    'ANTIQUE': 'Antique',
    'APAYAO': 'Apayao',
    'AURORA': 'Aurora',
    'BASILAN': 'Basilan',
    'BATAAN': 'Bataan',
    'BATANES': 'Batanes',
    'BATANGAS': 'Batangas',
    'BENGUET': 'Benguet',
    'BILIRAN': 'Biliran',
    'BOHOL': 'Bohol',
    'BUKIDNON': 'Bukidnon',
    'BULACAN': 'Bulacan',
    'CAGAYAN': 'Cagayan',
    'CAMARINES NORTE': 'Camarines Norte',
    'CAMARINES SUR': 'Camarines Sur',
    'CAM SUR': 'Camarines Sur', # Abbreviation found in soil data
    'CAMIGUIN': 'Camiguin',
    'CAPIZ': 'Capiz',
    'CATANDUANES': 'Catanduanes',
    'CAVITE': 'Cavite',
    'CEBU': 'Cebu',
    'CITY OF DAVAO': 'Davao', # Generalizing cities to province if possible or mapping to main province
    'CITY OF ZAMBOANGA': 'Zamboanga', # Generalizing
    'COMPOSTELA VALLEY': 'Davao de Oro', # Old name for Davao de Oro
    'COTABATO': 'Cotabato', # Note: could be North Cotabato
    'DAVAO OCCIDENTAL': 'Davao Occidental',
    'DAVAO ORIENTAL': 'Davao Oriental',
    'DAVAO DE ORO': 'Davao de Oro',
    'DAVAO DEL NORTE': 'Davao del Norte',
    'DAVAO DEL SUR': 'Davao del Sur',
    'DINAGAT ISLANDS': 'Dinagat Islands',
    'DINAGAT': 'Dinagat Islands', # Abbreviation
    'EASTERN SAMAR': 'Eastern Samar',
    'GENSAN': 'South Cotabato', # General Santos City, commonly associated with South Cotabato
    'GUIMARAS': 'Guimaras',
    'IFUGAO': 'Ifugao',
    'ILOCOS NORTE': 'Ilocos Norte',
    'ILOCOS SUR': 'Ilocos Sur',
    'ILOILO': 'Iloilo',
    'ISABELA': 'Isabela',
    'KALINGA': 'Kalinga',
    'LA UNION': 'La Union',
    'LAGUNA': 'Laguna',
    'LANAO DEL NORTE': 'Lanao del Norte',
    'LANAO DEL SUR': 'Lanao del Sur',
    'LEYTE': 'Leyte',
    'MAGUINDANAO': 'Maguindanao',
    'MARINDUQUE': 'Marinduque',
    'MASBATE': 'Masbate',
    'METRO MANILA': 'Metro Manila',
    'MISAMIS OCCIDENTAL': 'Misamis Occidental',
    'MISAMIS ORIENTAL': 'Misamis Oriental',
    'MOUNTAIN PROVINCE': 'Mountain Province',
    'MT PROVINCE': 'Mountain Province', # Abbreviation
    'NEGROS OCCIDENTAL': 'Negros Occidental',
    'NEGROS OCCIDENTAL A/': 'Negros Occidental', # Correcting potential parsing issues
    'NEGROS ORIENTAL': 'Negros Oriental',
    'NEGROS ORIENTAL B/': 'Negros Oriental', # Correcting potential parsing issues
    'NORTHERN SAMAR': 'Northern Samar',
    'NUEVA ECIJA': 'Nueva Ecija',
    'NUEVA VIZCAYA': 'Nueva Vizcaya',
    'OCCIDENTAL MINDORO': 'Occidental Mindoro',
    'ORIENTAL MINDORO': 'Oriental Mindoro',
    'PALAWAN': 'Palawan',
    'PAMPANGA': 'Pampanga',
    'PANGASINAN': 'Pangasinan',
    'QUEZON': 'Quezon',
    'QUIRINO': 'Quirino',
    'RIZAL': 'Rizal',
    'ROMBLON': 'Romblon',
    'SAMAR': 'Samar',
    'SARANGANI': 'Sarangani',
    'SIQUIJOR': 'Siquijor',
    'SIQUIJOR B/': 'Siquijor', # Correcting potential parsing issues
    'SORSOGON': 'Sorsogon',
    'SOUTH COTABATO': 'South Cotabato',
    'SOUTHERN LEYTE': 'Southern Leyte',
    'SULTAN KUDARAT': 'Sultan Kudarat',
    'SULU': 'Sulu',
    'SURIGAO DEL NORTE': 'Surigao del Norte',
    'SURIGAO DEL SUR': 'Surigao del Sur',
    'TARLAC': 'Tarlac',
    'TAWI-TAWI': 'Tawi-Tawi',
    'ZAMBALES': 'Zambales',
    'ZAMBOANGA SIBUGAY': 'Zamboanga Sibugay',
    'ZAMBOANGA DEL NORTE': 'Zamboanga del Norte',
    'ZAMBOANGA DEL SUR': 'Zamboanga del Sur',
    'ZAMBOANGA': 'Zamboanga', # Generic Zamboanga
}

def standardize_province_names(df, column_name='province'):
    """
    Standardizes province names in a DataFrame column using the PROVINCE_MAPPING.
    Converts to uppercase first for matching, then maps.
    """
    df[column_name] = df[column_name].astype(str).str.upper().map(PROVINCE_MAPPING).fillna(df[column_name])
    return df

def preprocess_soil_data(data_dir="raw_datasets"):
    """
    Loads and preprocesses soil test data for correlation analysis.
    - Standardizes province names.
    - Converts categorical NPK to numerical.
    - Calculates average pH.
    - Renames columns for consistency.
    - Aggregates to (province) level by taking the mean of numerical values.
    """
    soil_path = os.path.join(data_dir, "soil_test_data.csv")
    soil_df = pd.read_csv(soil_path)

    # Standardize province names early
    soil_df = standardize_province_names(soil_df, 'province')

    # Convert NPK requirements from categorical to numerical
    npk_mapping = {'low': 1, 'medium': 2, 'high': 3}
    for col in ['nitrogen', 'phosphorus', 'potassium']:
        soil_df[f'soil_{col}_level'] = soil_df[col].astype(str).str.lower().map(npk_mapping)

    # Calculate average pH
    soil_df['soil_ph_avg'] = (soil_df['ph_min'] + soil_df['ph_max']) / 2

    # Drop original NPK and pH columns, and farmer column
    soil_df = soil_df.drop(columns=['nitrogen', 'phosphorus', 'potassium', 'ph_min', 'ph_max', 'farmer'])
    
    # Rename for merging (lowercase province, municipality)
    soil_df = soil_df.rename(columns={'province': 'province', 'municipality': 'municipality'})

    # Aggregate to province level by taking the mean of numerical soil properties
    # This loses municipality level detail for soil, but necessary for broader merge
    soil_agg_province = soil_df.groupby('province').agg(
        avg_soil_nitrogen_level=('soil_nitrogen_level', 'mean'),
        avg_soil_phosphorus_level=('soil_phosphorus_level', 'mean'),
        avg_soil_potassium_level=('soil_potassium_level', 'mean'),
        avg_soil_ph_avg=('soil_ph_avg', 'mean')
    ).reset_index()

    return soil_agg_province

def preprocess_climate_data(data_dir="raw_datasets"):
    """
    Loads and preprocesses climate data, focusing on annual values for correlation.
    - Standardizes province names.
    - Pivots data to have annual values for each parameter as separate columns.
    - Renames columns for consistency.
    - Aggregates to (province, year) level by taking the mean across municipalities.
    """
    climate_path = os.path.join(data_dir, "climate_data.csv")
    climate_df = pd.read_csv(climate_path)

    # Standardize province names early
    climate_df = standardize_province_names(climate_df, 'Province')

    # Pivot the table to get PARAMETER values as columns
    climate_annual_df = climate_df.pivot_table(
        index=['Province', 'Municipality', 'YEAR'],
        columns='PARAMETER',
        values='ANNUAL'
    ).reset_index()

    # Rename columns for clarity
    climate_annual_df = climate_annual_df.rename(columns={
        'PRECTOTCORR_SUM': 'climate_annual_rainfall',
        'RH2M': 'climate_annual_humidity',
        'T2M': 'climate_annual_temperature'
    })
    
    # Rename for merging (lowercase province, municipality, year)
    climate_annual_df = climate_annual_df.rename(columns={'Province': 'province', 'Municipality': 'municipality', 'YEAR': 'year'})

    # Aggregate to province-year level by taking the mean across municipalities
    climate_agg_province_year = climate_annual_df.groupby(['province', 'year']).agg(
        avg_climate_annual_rainfall=('climate_annual_rainfall', 'mean'),
        avg_climate_annual_humidity=('climate_annual_humidity', 'mean'),
        avg_climate_annual_temperature=('climate_annual_temperature', 'mean')
    ).reset_index()

    return climate_agg_province_year

def preprocess_historical_performance_data(data_dir="raw_datasets"):
    """
    Loads and preprocesses historical crop performance data.
    - Standardizes province names.
    - Aggregates performance by Province, Year, Crop, and renames columns for consistency.
    """
    performance_path = os.path.join(data_dir, "historical_crop_performance.csv")
    performance_df = pd.read_csv(performance_path)

    # Standardize province names early
    performance_df = standardize_province_names(performance_df, 'Province')

    # Aggregate performance metrics by location and year (assuming per crop for now)
    agg_performance_df = performance_df.groupby(['Province', 'Year', 'Crop']).agg(
        total_volume_production=('Volume_Production', 'sum'),
        total_area_planted_harvested=('Area_Planted_Harvested', 'sum')
    ).reset_index()

    # Rename for merging (lowercase province, crop_name, year)
    agg_performance_df = agg_performance_df.rename(columns={'Province': 'province', 'Crop': 'crop_name', 'Year': 'year'})

    return agg_performance_df

def load_suitability_data(models_dir="models"):
    """
    Loads the suitability score data from the validation dataset used in model evaluation.
    - Standardizes province names.
    - Selects and renames key columns for merging.
    """
    val_df_path = os.path.join(models_dir, "validation_dataset.csv")
    try:
        val_df = pd.read_csv(val_df_path)
        # Standardize province names early
        val_df = standardize_province_names(val_df, 'province')
        
        # Select key columns and rename for consistency
        suitability_df = val_df[['province', 'crop_name', 'suitability_score']].copy()
        return suitability_df
    except FileNotFoundError:
        print(f"Error: {val_df_path} not found. Suitability score data cannot be loaded.")
        return pd.DataFrame() # Return empty DataFrame on error

def merge_all_data_for_correlation(data_dir="raw_datasets", models_dir="models"):
    """
    Merges preprocessed soil, climate, historical crop performance, and suitability data
    into a single DataFrame for correlation analysis.
    The granularity of the final DataFrame will be (province, year, crop_name).
    """
    soil_df_agg = preprocess_soil_data(data_dir) # aggregated to (province)
    climate_df_agg = preprocess_climate_data(data_dir) # aggregated to (province, year)
    performance_df = preprocess_historical_performance_data(data_dir) # (province, year, crop_name)
    suitability_df = load_suitability_data(models_dir) # (province, crop_name)

    print(f"\nUnique provinces in performance_df: {performance_df['province'].nunique()}")
    print(f"List of unique provinces in performance_df: {sorted(performance_df['province'].unique().tolist())}")
    print(f"\nUnique provinces in suitability_df: {suitability_df['province'].nunique()}")
    print(f"List of unique provinces in suitability_df: {sorted(suitability_df['province'].unique().tolist())}")
    print(f"\nUnique provinces in climate_df_agg: {climate_df_agg['province'].nunique()}")
    print(f"List of unique provinces in climate_df_agg: {sorted(climate_df_agg['province'].unique().tolist())}")
    print(f"\nUnique provinces in soil_df_agg: {soil_df_agg['province'].nunique()}")
    print(f"List of unique provinces in soil_df_agg: {sorted(soil_df_agg['province'].unique().tolist())}")

    # 1. Start with performance_df as the base
    merged_data = performance_df.copy()

    # 2. Merge suitability_df onto performance_df on (province, crop_name)
    merged_data = pd.merge(merged_data, suitability_df, on=['province', 'crop_name'], how='left')
    
    # 3. Merge climate_df onto merged_data on (province, year)
    merged_data = pd.merge(merged_data, climate_df_agg, on=['province', 'year'], how='left')

    # 4. Merge soil_df onto merged_data on (province)
    merged_data = pd.merge(merged_data, soil_df_agg, on=['province'], how='left')

    # Drop any rows with NaN values that resulted from merges
    initial_rows = merged_data.shape[0]
    merged_data = merged_data.dropna()
    rows_dropped = initial_rows - merged_data.shape[0]
    print(f"\nRows dropped due to NaNs after all merges: {rows_dropped}")
    print(f"Final shape after dropna: {merged_data.shape}")

    return merged_data

def extract_fertilizer_recommendation(nutrient_notes: str) -> str:
    """
    Extract fertilizer recommendation from nutrient notes field.
    
    Args:
        nutrient_notes: Nutrient notes from crop data
        
    Returns:
        Fertilizer recommendation string
    """
    if pd.isna(nutrient_notes) or not nutrient_notes:
        return "Apply balanced NPK fertilizer as per soil test results."
    return str(nutrient_notes).strip()


def parse_ph_range(ph_str: str) -> tuple:
    """
    Parse pH range string into (min, max) tuple.
    
    Args:
        ph_str: pH range string like "5.5-6.5" or "6.0"
        
    Returns:
        Tuple of (min_ph, max_ph)
    """
    if pd.isna(ph_str) or not ph_str:
        return (5.5, 7.5)  # Default neutral range
    
    ph_str = str(ph_str).strip()
    
    # Check for range format (e.g., "5.5-6.5")
    if '-' in ph_str:
        parts = ph_str.split('-')
        try:
            min_ph = float(parts[0].strip())
            max_ph = float(parts[1].strip())
            return (min_ph, max_ph)
        except (ValueError, IndexError):
            return (5.5, 7.5)
    
    # Single value
    try:
        ph = float(ph_str)
        return (ph, ph)
    except ValueError:
        return (5.5, 7.5)


def parse_soil_types(soil_type_str: str) -> list:
    """
    Parse soil type string into list of acceptable soil types.
    
    Args:
        soil_type_str: Soil type string like "Loam, Clay" or "Sandy"
        
    Returns:
        List of soil type strings
    """
    if pd.isna(soil_type_str) or not soil_type_str:
        return []
    
    soil_type_str = str(soil_type_str).strip()
    
    # Split by common delimiters
    for delimiter in [',', ';', '/']:
        if delimiter in soil_type_str:
            return [s.strip() for s in soil_type_str.split(delimiter) if s.strip()]
    
    return [soil_type_str]


def normalize_npk_level(level: str) -> str:
    """
    Normalize NPK level string to standard format.
    
    Args:
        level: NPK level string (e.g., "low", "LOW", "Low", "L")
        
    Returns:
        Normalized string: "Low", "Medium", or "High"
    """
    if pd.isna(level) or not level:
        return "Medium"
    
    level_str = str(level).strip().lower()
    
    # Map variations
    if level_str in ['low', 'l', 'lo']:
        return "Low"
    elif level_str in ['medium', 'med', 'm', 'moderate']:
        return "Medium"
    elif level_str in ['high', 'h', 'hi']:
        return "High"
    
    return "Medium"  # Default


def parse_planting_period(period_str: str) -> tuple:
    """
    Parse planting period string into (start_month, end_month) tuple.
    
    Args:
        period_str: Planting period string like "March-May" or "All season"
        
    Returns:
        Tuple of (start_month, end_month) as integers (1-12), or None if all season
    """
    if pd.isna(period_str) or not period_str:
        return None  # All season
    
    period_str = str(period_str).strip().lower()
    
    # Check for all season
    if 'all' in period_str or 'year-round' in period_str or 'any' in period_str:
        return None
    
    # Month name mapping
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    # Try to parse range (e.g., "March-May", "Mar-May")
    for delimiter in ['-', 'to', '–']:
        if delimiter in period_str:
            parts = period_str.split(delimiter)
            if len(parts) >= 2:
                start = parts[0].strip()
                end = parts[1].strip()
                
                # Find month numbers
                start_month = None
                end_month = None
                
                for month_name, month_num in month_map.items():
                    if month_name in start:
                        start_month = month_num
                    if month_name in end:
                        end_month = month_num
                
                if start_month and end_month:
                    return (start_month, end_month)
    
    return None  # Can't parse, assume all season


def is_in_planting_season(planting_period: tuple, current_month: int) -> bool:
    """
    Check if current month is within planting season.
    
    Args:
        planting_period: Tuple of (start_month, end_month)
        current_month: Current month (1-12)
        
    Returns:
        True if in season, False otherwise
    """
    if planting_period is None:
        return True  # All season
    
    start_month, end_month = planting_period
    
    # Handle wrapping (e.g., Nov-Jan)
    if start_month <= end_month:
        return start_month <= current_month <= end_month
    else:
        return current_month >= start_month or current_month <= end_month


def calculate_distance_to_season(current_month: int, start_month: int, end_month: int) -> int:
    """
    Calculate months until next planting season.
    
    Args:
        current_month: Current month (1-12)
        start_month: Season start month (1-12)
        end_month: Season end month (1-12)
        
    Returns:
        Number of months until season starts
    """
    if start_month <= end_month:
        # Normal range (e.g., Mar-May)
        if current_month < start_month:
            return start_month - current_month
        elif current_month > end_month:
            return (12 - current_month) + start_month
        else:
            return 0  # Currently in season
    else:
        # Wrapping range (e.g., Nov-Jan)
        if start_month <= current_month <= 12:
            return 0  # Currently in season
        elif 1 <= current_month <= end_month:
            return 0  # Currently in season
        else:
            return start_month - current_month

def parse_yield_range(yield_str: str) -> tuple:
    """
    Parse yield range string into (min, max) tuple.
    
    Args:
        yield_str: Yield string like "4.5-6.0 tons/ha" or "5.0"
        
    Returns:
        Tuple of (min_yield, max_yield) in tons/ha
    """
    if pd.isna(yield_str) or not yield_str:
        return (0.0, 0.0)
    
    yield_str = str(yield_str).strip().lower()
    
    # Remove units
    yield_str = yield_str.replace('tons/ha', '').replace('ton/ha', '').replace('t/ha', '').strip()
    
    # Check for range format (e.g., "4.5-6.0")
    if '-' in yield_str:
        parts = yield_str.split('-')
        try:
            min_yield = float(parts[0].strip())
            max_yield = float(parts[1].strip())
            return (min_yield, max_yield)
        except (ValueError, IndexError):
            return (0.0, 0.0)
    
    # Single value
    try:
        val = float(yield_str)
        return (val, val)
    except ValueError:
        return (0.0, 0.0)



if __name__ == "__main__":
    # Example usage for correlation matrix data preparation
    merged_data = merge_all_data_for_correlation()
    print("Merged Data Head:")
    print(merged_data.head())
    print("\nMerged Data Info:")
    merged_data.info()
    print("\nMerged Data Describe:")
    merged_data.describe()