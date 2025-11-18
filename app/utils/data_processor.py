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

if __name__ == "__main__":
    # Example usage for correlation matrix data preparation
    merged_data = merge_all_data_for_correlation()
    print("Merged Data Head:")
    print(merged_data.head())
    print("\nMerged Data Info:")
    merged_data.info()
    print("\nMerged Data Describe:")
    merged_data.describe()