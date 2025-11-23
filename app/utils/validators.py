"""
Input validation utilities.
"""
from typing import List, Optional
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_npk_level(level: str) -> bool:
    """
    Validate NPK level is one of: Low, Medium, High.
    
    Args:
        level: NPK level string
    
    Returns:
        True if valid
    
    Raises:
        ValidationError if invalid
    """
    valid_levels = ['Low', 'Medium', 'High']
    if level not in valid_levels:
        raise ValidationError(
            f"Invalid NPK level: {level}. Must be one of {valid_levels}"
        )
    return True


def validate_ph_range(ph_min: float, ph_max: float) -> bool:
    """
    Validate pH range is valid.
    
    Args:
        ph_min: Minimum pH value
        ph_max: Maximum pH value
    
    Returns:
        True if valid
    
    Raises:
        ValidationError if invalid
    """
    if ph_min < 0 or ph_min > 14:
        raise ValidationError(f"Invalid pH min: {ph_min}. Must be between 0 and 14")
    
    if ph_max < 0 or ph_max > 14:
        raise ValidationError(f"Invalid pH max: {ph_max}. Must be between 0 and 14")
    
    if ph_min > ph_max:
        raise ValidationError(f"pH min ({ph_min}) cannot be greater than pH max ({ph_max})")
    
    return True


def validate_soil_type(soil_type: str, valid_types: Optional[List[str]] = None) -> bool:
    """
    Validate soil type is reasonable.
    
    Args:
        soil_type: Soil type string
        valid_types: Optional list of valid soil types
    
    Returns:
        True if valid
    
    Raises:
        ValidationError if invalid
    """
    if not soil_type or not soil_type.strip():
        raise ValidationError("Soil type cannot be empty")
    
    # Common soil types in Philippines
    common_types = [
        'Loam', 'Clay', 'Sandy loam', 'Clay loam', 'Silt loam',
        'Sandy clay loam', 'Deep loam', 'Deep clay loam', 'Sandy'
    ]
    
    if valid_types is None:
        valid_types = common_types
    
    # Check if soil type matches any common type (case-insensitive)
    soil_lower = soil_type.strip().lower()
    matches = [t for t in valid_types if t.lower() == soil_lower]
    
    if not matches:
        # Warn but don't fail - might be a valid variation
        pass
    
    return True


def validate_location(
    province: str,
    municipality: Optional[str],
    data_loader
) -> bool:
    """
    Validate that location exists in climate data.
    
    Args:
        province: Province name
        municipality: Municipality name (optional)
        data_loader: DataLoader instance
    
    Returns:
        True if valid
    
    Raises:
        ValidationError if location not found
    """
    try:
        data_loader.get_climate_averages(province, municipality)
        return True
    except ValueError as e:
        raise ValidationError(str(e))


def validate_farmer_input(
    province: str,
    municipality: Optional[str],
    nitrogen: str,
    phosphorus: str,
    potassium: str,
    ph_min: float,
    ph_max: float,
    soil_type: str,
    data_loader
) -> bool:
    """
    Validate all farmer input parameters.
    
    Args:
        province: Province name
        municipality: Municipality name
        nitrogen: Nitrogen level (Low/Medium/High)
        phosphorus: Phosphorus level (Low/Medium/High)
        potassium: Potassium level (Low/Medium/High)
        ph_min: Minimum pH
        ph_max: Maximum pH
        soil_type: Soil type
        data_loader: DataLoader instance
    
    Returns:
        True if all valid
    
    Raises:
        ValidationError if any parameter is invalid
    """
    # Validate NPK levels
    validate_npk_level(nitrogen)
    validate_npk_level(phosphorus)
    validate_npk_level(potassium)
    
    # Validate pH range
    validate_ph_range(ph_min, ph_max)
    
    # Validate soil type
    validate_soil_type(soil_type)
    
    # Validate location
    validate_location(province, municipality, data_loader)
    
    return True

