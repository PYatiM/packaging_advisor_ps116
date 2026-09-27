from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


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
    quantity: int = Field(default=1, ge=1, le=200)
    declared_value_inr: Optional[float] = Field(default=None, ge=0.0)
    requires_cold_chain: bool = False


class CartRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: List[SKUItem]
    source_pin: str = "110001"
    destination_pin: str
    use_genai: bool = False
    model_provider: str = "Gemini"
    model_name: Optional[str] = None
    optimization_priority: Literal[
        "balanced", "lowest_cost", "lowest_risk", "eco_friendly"
    ] = "balanced"
    customer_or_po: str = ""
    persist_history: bool = True


class ItemPlacement(BaseModel):
    sku_id: str
    product_category: str
    x_cm: float
    y_cm: float
    z_cm: float
    dx_cm: float
    dy_cm: float
    dz_cm: float
    pad_cm: float = 0.0
    weight_kg: float
    fragility_score: float
    bin_index: int = 0


class BoxAlternative(BaseModel):
    tier_name: str
    carton_id: str
    total_cost: float
    utilization_pct: float
    risk_probability: float
    cost_breakdown: str
    box_count: int = 1
    transit_mode: str = ""
    estimated_days: str = ""
    expected_damage_cost: float = 0.0
    effective_landed_cost: float = 0.0
    board_spec: str = ""
    co2_kg: float = 0.0


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
    group_label: str = ""
    box_count: int = 1
    container_inner_dims_cm: List[float] = []
    container_spec_summary: str = ""
    net_utilization_pct: float = 0.0
    padded_utilization_pct: float = 0.0
    chargeable_weight_kg: float = 0.0
    actual_gross_weight_kg: float = 0.0
    volumetric_weight_kg: float = 0.0
    declared_value_inr: float = 0.0
    expected_damage_cost: float = 0.0
    estimated_transit_days: str = ""
    placements: List[ItemPlacement] = []
    shipment_db_id: Optional[int] = None
    transit_mode: str = ""
    co2_emissions_kg: float = 0.0
    handling_badges: List[str] = []


class RecommendationResponse(BaseModel):
    shipments: List[ShipmentResponse]
    total_cart_cost: float
    compatibility_warning: str = ""
    source_zone: str = ""
    destination_zone: str = ""
    shipping_zone_tier: str = ""
    total_expected_damage_cost: float = 0.0
    total_effective_landed_cost: float = 0.0
    grouping_reasons: List[str] = []
    order_ref: str = ""
    baseline_unoptimized_cost: float = 0.0
    cost_savings_inr: float = 0.0
    cost_savings_pct: float = 0.0
    total_co2_kg: float = 0.0
    co2_saved_kg: float = 0.0
