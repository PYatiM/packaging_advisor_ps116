from pydantic import BaseModel, Field
from typing import List

class SKUItem(BaseModel):
    sku_id: str
    product_category: str = "General"
    length_cm: float = Field(..., gt=0)
    width_cm: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    fragility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_liquid: bool = False
    orientation_sensitive: bool = False

class CartRequest(BaseModel):
    items: List[SKUItem]
    source_pin: str = "110001"
    destination_pin: str
    use_genai: bool = False
    model_provider: str = "Gemini"
    
class BoxAlternative(BaseModel):
    tier_name: str
    carton_id: str
    total_cost: float
    utilization_pct: float
    risk_probability: float
    cost_breakdown: str

class ShipmentResponse(BaseModel):
    items: List[SKUItem]
    recommended_carton_id: str
    total_estimated_cost: float
    damage_risk_probability: float
    genai_explanation: str
    carton_reasoning: str = ""
    cost_breakdown: str = ""
    risk_reasoning: str = ""
    packing_instructions: str = ""
    transit_mode_advice: str = ""
    alternatives: List[BoxAlternative] = []

class RecommendationResponse(BaseModel):
    shipments: List[ShipmentResponse]
    total_cart_cost: float
    compatibility_warning: str = ""
    source_zone: str = ""
    destination_zone: str = ""
