"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum


class NPKLevel(str, Enum):
    """NPK level enumeration."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RecommendationRequest(BaseModel):
    """Request schema for crop recommendation."""
    province: str = Field(..., description="Province name")
    municipality: str = Field(..., description="Municipality name")
    nitrogen: NPKLevel = Field(..., description="Nitrogen level (Low/Medium/High)")
    phosphorus: NPKLevel = Field(..., description="Phosphorus level (Low/Medium/High)")
    potassium: NPKLevel = Field(..., description="Potassium level (Low/Medium/High)")
    ph_min: float = Field(..., ge=0, le=14, description="Minimum pH value (0-14)")
    ph_max: float = Field(..., ge=0, le=14, description="Maximum pH value (0-14)")
    soil_type: str = Field(..., description="Soil type (e.g., Loam, Clay, Sandy loam)")
    
    @validator('ph_max')
    def ph_max_greater_than_min(cls, v, values):
        """Validate that ph_max is greater than ph_min."""
        if 'ph_min' in values and v < values['ph_min']:
            raise ValueError('ph_max must be greater than or equal to ph_min')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "province": "Cagayan",
                "municipality": "Tuguegarao",
                "nitrogen": "Medium",
                "phosphorus": "High",
                "potassium": "Medium",
                "ph_min": 6.0,
                "ph_max": 6.8,
                "soil_type": "Loam"
            }
        }


class RuleBreakdown(BaseModel):
    """Rule-based score breakdown."""
    npk: float
    ph: float
    temperature: float
    rainfall: float
    humidity: float
    soil_type: float


class MLBreakdown(BaseModel):
    """ML model score breakdown."""
    npk_match: float
    ph_proximity: float
    temp_suitability: float
    rainfall_suitability: float
    humidity_suitability: float
    soil_match: float
    historical_yield: float
    season_alignment: float
    regional_success: float


class CropRecommendation(BaseModel):
    """Individual crop recommendation."""
    rank: int
    crop_name: str
    category: str
    hybrid_score: float
    rule_score: float
    ml_model_score: float
    confidence: float
    badge: Optional[str] = None
    expected_yield: str
    risks: List[str]
    planting_season: str
    days_to_harvest: str
    fertilizer_recommendation: str
    why_recommended: str
    rule_breakdown: RuleBreakdown
    ml_breakdown: MLBreakdown


class LocationSummary(BaseModel):
    """Location summary."""
    province: str
    municipality: str


class ClimateSummary(BaseModel):
    """Climate summary."""
    avg_temperature: Optional[float] = None
    avg_rainfall: Optional[float] = None
    avg_humidity: Optional[float] = None


class SoilSummary(BaseModel):
    """Soil summary."""
    nitrogen: str
    phosphorus: str
    potassium: str
    ph_range: str
    soil_type: str


class Insights(BaseModel):
    """Recommendation insights."""
    best_season: Optional[str] = None
    soil_amendments: Optional[str] = None
    diversification_tip: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response schema for crop recommendation."""
    location: LocationSummary
    climate_summary: ClimateSummary
    soil_summary: SoilSummary
    recommendations: List[CropRecommendation]
    insights: Insights
    
    class Config:
        schema_extra = {
            "example": {
                "location": {
                    "province": "Cagayan",
                    "municipality": "Tuguegarao"
                },
                "climate_summary": {
                    "avg_temperature": 27.5,
                    "avg_rainfall": 1850.0,
                    "avg_humidity": 78.0
                },
                "soil_summary": {
                    "nitrogen": "Medium",
                    "phosphorus": "High",
                    "potassium": "Medium",
                    "ph_range": "6.0-6.8",
                    "soil_type": "Loam"
                },
                "recommendations": [],
                "insights": {}
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None

