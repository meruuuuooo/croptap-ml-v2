"""
Data processing utilities for parsing and normalizing crop data.
"""
import re
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime


def parse_ph_range(ph_range_str: str) -> Tuple[float, float]:
    """
    Parse pH range string (e.g., "6.0-7.0") into min and max values.
    
    Args:
        ph_range_str: String like "6.0-7.0" or "5.5-6.5"
    
    Returns:
        Tuple of (min_ph, max_ph)
    """
    if pd.isna(ph_range_str) or not ph_range_str:
        return (0.0, 14.0)  # Default to full range if not specified
    
    # Extract numbers from string
    numbers = re.findall(r'\d+\.?\d*', str(ph_range_str))
    
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    elif len(numbers) == 1:
        # Single value, assume small range
        val = float(numbers[0])
        return (val - 0.5, val + 0.5)
    else:
        return (0.0, 14.0)


def parse_soil_types(soil_type_str: str) -> List[str]:
    """
    Parse soil type string into list of acceptable types.
    
    Args:
        soil_type_str: String like "Loam, Clay loam, Sandy loam" or "Sandy loam"
    
    Returns:
        List of soil type strings
    """
    if pd.isna(soil_type_str) or not soil_type_str:
        return []
    
    # Split by comma and clean
    types = [t.strip() for t in str(soil_type_str).split(',')]
    # Remove empty strings
    types = [t for t in types if t]
    return types


def parse_yield_range(yield_str: str) -> Tuple[float, float]:
    """
    Parse yield range string (e.g., "2-4 tons" or "15-20 tons") into min and max.
    
    Args:
        yield_str: String like "2-4 tons" or "15-20 tons"
    
    Returns:
        Tuple of (min_yield, max_yield) in tons/ha
    """
    if pd.isna(yield_str) or not yield_str:
        return (0.0, 0.0)
    
    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', str(yield_str))
    
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    elif len(numbers) == 1:
        val = float(numbers[0])
        return (val, val)
    else:
        return (0.0, 0.0)


def parse_planting_period(planting_period_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse planting period string into start and end months.
    
    Args:
        planting_period_str: String like "May-Aug", "Oct-Mar", "All season"
    
    Returns:
        Tuple of (start_month, end_month) where months are 1-12, or None for "All season"
    """
    if pd.isna(planting_period_str) or not planting_period_str:
        return None
    
    period = str(planting_period_str).strip()
    
    if "all season" in period.lower():
        return None  # None indicates all season
    
    # Month name to number mapping
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    # Try to extract month names
    months = re.findall(r'[a-z]+', period.lower())
    
    if len(months) >= 2:
        start_month = month_map.get(months[0])
        end_month = month_map.get(months[-1])
        
        if start_month and end_month:
            return (start_month, end_month)
    
    return None


def is_in_planting_season(
    planting_period: Optional[Tuple[int, int]], 
    current_month: Optional[int] = None
) -> bool:
    """
    Check if current month is within planting period.
    
    Args:
        planting_period: Tuple of (start_month, end_month) or None for all season
        current_month: Current month (1-12). If None, uses current date.
    
    Returns:
        True if in planting season, False otherwise
    """
    if planting_period is None:
        return True  # All season
    
    if current_month is None:
        current_month = datetime.now().month
    
    start_month, end_month = planting_period
    
    # Handle year wrap (e.g., Oct-Mar)
    if start_month > end_month:
        return current_month >= start_month or current_month <= end_month
    else:
        return start_month <= current_month <= end_month


def calculate_distance_to_season(
    current_month: int,
    start_month: int,
    end_month: int
) -> int:
    """
    Calculate months until next planting window.
    
    Args:
        current_month: Current month (1-12)
        start_month: Planting season start month
        end_month: Planting season end month
    
    Returns:
        Number of months until next planting window
    """
    # Handle year wrap
    if start_month > end_month:
        # Season spans year boundary (e.g., Oct-Mar)
        if current_month >= start_month or current_month <= end_month:
            return 0  # Currently in season
        elif current_month > end_month:
            return start_month - current_month
        else:
            return start_month - current_month
    else:
        # Normal season (e.g., May-Aug)
        if start_month <= current_month <= end_month:
            return 0  # Currently in season
        elif current_month < start_month:
            return start_month - current_month
        else:
            # After season, calculate to next year
            return (12 - current_month) + start_month


def normalize_npk_level(level: str) -> str:
    """
    Normalize NPK level string to standard format.
    
    Args:
        level: NPK level string (case-insensitive)
    
    Returns:
        Normalized level: "Low", "Medium", or "High"
    """
    if not level:
        return "Medium"  # Default
    
    level_lower = str(level).strip().lower()
    
    if level_lower in ['low', 'l']:
        return "Low"
    elif level_lower in ['medium', 'med', 'm']:
        return "Medium"
    elif level_lower in ['high', 'h']:
        return "High"
    else:
        return "Medium"  # Default to medium if unclear


def extract_fertilizer_recommendation(nutrient_notes: str) -> str:
    """
    Extract fertilizer recommendation from nutrient notes.
    
    Args:
        nutrient_notes: JSON string or text with nutrient information
    
    Returns:
        Formatted fertilizer recommendation string
    """
    if pd.isna(nutrient_notes) or not nutrient_notes:
        return "Follow standard NPK recommendations for this crop type."
    
    # Try to extract key information
    notes = str(nutrient_notes)
    
    # Look for specific recommendations
    if "apply" in notes.lower() or "kg" in notes.lower():
        # Extract sentences with recommendations
        sentences = re.split(r'[.!?]', notes)
        recommendations = [s.strip() for s in sentences if 'apply' in s.lower() or 'kg' in s.lower()]
        if recommendations:
            return recommendations[0] + "."
    
    return "Follow standard NPK recommendations for this crop type."

