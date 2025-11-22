"""
Main recommendation engine.
Combines rule-based and ML model scores to generate recommendations.
"""
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from app.services.data_loader import get_data_loader
from app.services.rule_based_scorer import RuleBasedScorer
from app.services.feature_extractor import FeatureExtractor
from app.services.ml_model import get_ml_model_service
from app.services.yield_predictor import YieldPredictor
from app.utils.data_processor import extract_fertilizer_recommendation


class RecommendationEngine:
    """Main recommendation engine combining rule-based and ML model scoring."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.data_loader = get_data_loader()
        self.rule_scorer = RuleBasedScorer()
        self.feature_extractor = FeatureExtractor()
        self.ml_model = get_ml_model_service()
        self.yield_predictor = YieldPredictor()
    
    def generate_recommendations(
        self,
        province: str,
        municipality: str,
        nitrogen: str,
        phosphorus: str,
        potassium: str,
        ph_min: float,
        ph_max: float,
        soil_type: str,
        top_n: int = 10
    ) -> Dict:
        """
        Generate top N crop recommendations.
        
        Args:
            province: Province name
            municipality: Municipality name
            nitrogen: Nitrogen level (Low/Medium/High)
            phosphorus: Phosphorus level (Low/Medium/High)
            potassium: Potassium level (Low/Medium/High)
            ph_min: Minimum pH
            ph_max: Maximum pH
            soil_type: Soil type
            top_n: Number of top recommendations to return
        
        Returns:
            Dict with recommendations and metadata
        """
        # Get climate data
        climate = self.data_loader.get_climate_averages(province, municipality)
        avg_temp = climate['temperature']
        avg_rainfall = climate['rainfall']
        avg_humidity = climate['humidity']
        
        # Get all crops
        all_crops = self.data_loader.get_all_crops()
        
        # Current month for season alignment
        current_month = datetime.now().month
        
        recommendations = []
        
        # Loop through all crops
        for idx, crop_row in all_crops.iterrows():
            crop_name = crop_row['Crop_Name']
            
            # Get historical yield data
            historical_data = self.data_loader.get_historical_yield(
                crop_name, province
            )
            
            # Calculate rule-based score
            rule_result = self.rule_scorer.calculate_score(
                crop_data=crop_row,
                farmer_nitrogen=nitrogen,
                farmer_phosphorus=phosphorus,
                farmer_potassium=potassium,
                farmer_ph_min=ph_min,
                farmer_ph_max=ph_max,
                farmer_soil_type=soil_type,
                avg_temperature=avg_temp,
                avg_rainfall=avg_rainfall,
                avg_humidity=avg_humidity
            )
            rule_score = rule_result['total_score']
            rule_breakdown = rule_result['breakdown']
            
            # Extract features once (used by both ML model and yield prediction)
            features = self.feature_extractor.extract_features(
                crop_data=crop_row,
                farmer_nitrogen=nitrogen,
                farmer_phosphorus=phosphorus,
                farmer_potassium=potassium,
                farmer_ph_min=ph_min,
                farmer_ph_max=ph_max,
                farmer_soil_type=soil_type,
                avg_temperature=avg_temp,
                avg_rainfall=avg_rainfall,
                avg_humidity=avg_humidity,
                historical_yield_data=historical_data,
                current_month=current_month
            )
            
            # Calculate ML model score (pass pre-computed features)
            ml_model_score = self.ml_model.predict_score(
                crop_data=crop_row,
                farmer_nitrogen=nitrogen,
                farmer_phosphorus=phosphorus,
                farmer_potassium=potassium,
                farmer_ph_min=ph_min,
                farmer_ph_max=ph_max,
                farmer_soil_type=soil_type,
                avg_temperature=avg_temp,
                avg_rainfall=avg_rainfall,
                avg_humidity=avg_humidity,
                historical_yield_data=historical_data,
                current_month=current_month,
                province=province,
                crop_category=crop_row.get('Category', ''),
                features=features  # Pass pre-computed features
            )
            
            # Calculate hybrid score (40% rule-based, 60% ML model)
            hybrid_score = (
                rule_score * 0.40 +
                ml_model_score * 0.60
            )
            
            # Calculate confidence (agreement between rule-based and ML model)
            confidence = max(0.0, 100.0 - abs(rule_score - ml_model_score))
            
            # Identify risks
            risks = self._identify_risks(rule_breakdown, historical_data)
            
            # Predict expected yield
            climate_factor = self.yield_predictor.calculate_climate_factor(
                features['temp_suitability'],
                features['rainfall_suitability'],
                features['humidity_suitability']
            )
            soil_factor = self.yield_predictor.calculate_soil_factor(
                features['npk_match'],
                features['ph_proximity'],
                features['soil_match']
            )
            expected_yield = self.yield_predictor.predict_yield(
                crop_data=crop_row,
                historical_yield_data=historical_data,
                climate_factor=climate_factor,
                soil_factor=soil_factor
            )
            
            # Generate "why recommended" text
            why_recommended = self._generate_why_recommended(
                crop_row, rule_breakdown, historical_data, province
            )
            
            # Extract fertilizer recommendation
            nutrient_notes = crop_row.get('nutrient_notes', '')
            fertilizer_rec = extract_fertilizer_recommendation(nutrient_notes)
            
            recommendation = {
                'crop_name': crop_name,
                'category': crop_row.get('Category', 'Unknown'),
                'hybrid_score': round(hybrid_score, 2),
                'rule_score': round(rule_score, 2),
                'ml_model_score': round(ml_model_score, 2),
                'confidence': round(confidence, 2),
                'expected_yield': expected_yield,
                'risks': risks,
                'planting_season': crop_row.get('Planting Period', 'Unknown'),
                'days_to_harvest': crop_row.get('Days to Harvest', 'Unknown'),
                'fertilizer_recommendation': fertilizer_rec,
                'why_recommended': why_recommended,
                'rule_breakdown': {
                    'npk': round(rule_breakdown['npk'], 2),
                    'ph': round(rule_breakdown['ph'], 2),
                    'temperature': round(rule_breakdown['temperature'], 2),
                    'rainfall': round(rule_breakdown['rainfall'], 2),
                    'humidity': round(rule_breakdown['humidity'], 2),
                    'soil_type': round(rule_breakdown['soil_type'], 2)
                },
                'ml_breakdown': {
                    'npk_match': round(features['npk_match'], 2),
                    'ph_proximity': round(features['ph_proximity'], 2),
                    'temp_suitability': round(features['temp_suitability'], 2),
                    'rainfall_suitability': round(features['rainfall_suitability'], 2),
                    'humidity_suitability': round(features['humidity_suitability'], 2),
                    'soil_match': round(features['soil_match'], 2),
                    'historical_yield': round(features['historical_yield'], 2),
                    'season_alignment': round(features['season_alignment'], 2),
                    'regional_success': round(features['regional_success'], 2)
                }
            }
            
            recommendations.append(recommendation)
        
        # Sort by hybrid score (descending)
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Take top N and assign ranks/badges
        top_recommendations = recommendations[:top_n]
        for i, rec in enumerate(top_recommendations):
            rec['rank'] = i + 1
            if i == 0:
                rec['badge'] = "Best Match"
            elif i == 1:
                rec['badge'] = "Second Best"
            elif i == 2:
                rec['badge'] = "Third Best"
            else:
                rec['badge'] = None
        
        return {
            'location': {
                'province': province,
                'municipality': municipality
            },
            'climate_summary': {
                'avg_temperature': round(avg_temp, 2) if avg_temp else None,
                'avg_rainfall': round(avg_rainfall, 2) if avg_rainfall else None,
                'avg_humidity': round(avg_humidity, 2) if avg_humidity else None
            },
            'soil_summary': {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'ph_range': f"{ph_min}-{ph_max}",
                'soil_type': soil_type
            },
            'recommendations': top_recommendations,
            'insights': self._generate_insights(top_recommendations, climate)
        }
    
    def _identify_risks(
        self,
        rule_breakdown: Dict,
        historical_data: Optional[Dict]
    ) -> List[str]:
        """Identify risks based on low component scores."""
        risks = []
        
        if rule_breakdown['npk'] < 20:
            risks.append("NPK mismatch - fertilizer needed")
        
        if rule_breakdown['ph'] < 15:
            risks.append("pH out of range - soil amendment required")
        
        if rule_breakdown['temperature'] < 15:
            risks.append("Temperature not ideal - timing critical")
        
        if rule_breakdown['rainfall'] < 10:
            risks.append("Rainfall insufficient - irrigation needed")
        
        # Check historical data for limited success
        if historical_data is None:
            risks.append("Limited success history in this province")
        elif historical_data.get('avg_yield_per_ha', 0) < 6.0:  # Less than 6 tons/ha
            risks.append("Limited success history in this province")
        
        return risks
    
    def _generate_why_recommended(
        self,
        crop_data: pd.Series,
        rule_breakdown: Dict,
        historical_data: Optional[Dict],
        province: str
    ) -> str:
        """Generate explanation text for why crop is recommended."""
        reasons = []
        
        # Check top scoring factors from rule-based
        if rule_breakdown['npk'] >= 25:
            reasons.append("excellent NPK match")
        
        if rule_breakdown['ph'] >= 18:
            reasons.append("ideal pH range")
        
        if rule_breakdown['temperature'] >= 18:
            reasons.append("suitable temperature conditions")
        
        if rule_breakdown['rainfall'] >= 12:
            reasons.append("adequate rainfall")
        
        # Add historical performance context (ML model considers this)
        if historical_data:
            avg_yield = historical_data.get('avg_yield_per_ha', 0)
            years = historical_data.get('years_of_data', 0)
            reasons.append(
                f"proven success in {province} with avg {avg_yield:.1f} tons/ha over {years} years"
            )
        
        if not reasons:
            reasons.append("moderate suitability across all factors")
        
        return " ".join(reasons).capitalize() + "."
    
    def _generate_insights(
        self,
        recommendations: List[Dict],
        climate: Dict
    ) -> Dict[str, str]:
        """Generate insights based on recommendations."""
        insights = {}
        
        # Analyze top recommendations
        if recommendations:
            top_crop = recommendations[0]
            category = top_crop.get('category', '')
            
            # Best season
            planting_season = top_crop.get('planting_season', '')
            if planting_season and 'all season' not in planting_season.lower():
                insights['best_season'] = f"{planting_season} for optimal planting"
            else:
                insights['best_season'] = "Year-round planting possible"
            
            # Soil amendments
            risks_count = sum(len(r.get('risks', [])) for r in recommendations[:3])
            if risks_count > 0:
                insights['soil_amendments'] = "Consider soil amendments for optimal yields"
            else:
                insights['soil_amendments'] = "Soil conditions are suitable for most crops"
            
            # Diversification tip
            if len(recommendations) >= 2:
                crop1 = recommendations[0]['crop_name']
                crop2 = recommendations[1]['crop_name']
                insights['diversification_tip'] = f"Consider mixing {crop1} with {crop2} for crop rotation"
        
        return insights

