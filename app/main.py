"""
FastAPI application for crop recommendation system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ErrorResponse
)
from app.services.recommendation_engine import RecommendationEngine
from app.utils.validators import validate_farmer_input, ValidationError
from app.services.data_loader import get_data_loader
import traceback

app = FastAPI(
    title="Crop Recommendation System API",
    description="Hybrid crop recommendation system combining rule-based and ML approaches",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
recommendation_engine = RecommendationEngine()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CropTAP - Crop Recommendation System using Data-Driven Algorithm API",
        "version": "2.0.0",
        "endpoints": {
            "POST /recommend": "Get crop recommendations",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        data_loader = get_data_loader()
        crop_count = len(data_loader.get_all_crops())
        return {
            "status": "healthy",
            "crops_loaded": crop_count,
            "ml_model_loaded": recommendation_engine.ml_model.is_loaded
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Validate input
        data_loader = get_data_loader()
        validate_farmer_input(
            province=request.province,
            municipality=request.municipality,
            nitrogen=request.nitrogen.value,
            phosphorus=request.phosphorus.value,
            potassium=request.potassium.value,
            ph_min=request.ph_min,
            ph_max=request.ph_max,
            soil_type=request.soil_type,
            data_loader=data_loader
        )
        
        # Generate recommendations
        result = recommendation_engine.generate_recommendations(
            province=request.province,
            municipality=request.municipality,
            nitrogen=request.nitrogen.value,
            phosphorus=request.phosphorus.value,
            potassium=request.potassium.value,
            ph_min=request.ph_min,
            ph_max=request.ph_max,
            soil_type=request.soil_type,
            top_n=10
        )
        
        return RecommendationResponse(**result)
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

