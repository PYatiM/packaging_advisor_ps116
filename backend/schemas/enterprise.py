"""Pydantic schemas for Enterprise DB endpoints (SKUs, Custom Boxes, Orders, Analytics)."""

from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class MasterSKUCreate(BaseModel):
    sku_code: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=160)
    product_category: str = "General"
    length_cm: float = Field(..., gt=0)
    width_cm: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    fragility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    declared_value_inr: Optional[float] = Field(default=None, ge=0.0)
    is_liquid: bool = False
    orientation_sensitive: bool = False
    requires_cold_chain: bool = False


class MasterSKUUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=160)
    product_category: Optional[str] = None
    length_cm: Optional[float] = Field(default=None, gt=0)
    width_cm: Optional[float] = Field(default=None, gt=0)
    height_cm: Optional[float] = Field(default=None, gt=0)
    weight_kg: Optional[float] = Field(default=None, gt=0)
    fragility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    declared_value_inr: Optional[float] = Field(default=None, ge=0.0)
    is_liquid: Optional[bool] = None
    orientation_sensitive: Optional[bool] = None
    requires_cold_chain: Optional[bool] = None


class MasterSKUResponse(BaseModel):
    id: int
    sku_code: str
    name: str
    product_category: str
    length_cm: float
    width_cm: float
    height_cm: float
    weight_kg: float
    fragility_score: float
    declared_value_inr: float
    is_liquid: bool
    orientation_sensitive: bool
    requires_cold_chain: bool
    created_at: datetime
    updated_at: datetime


class CustomContainerCreate(BaseModel):
    container_id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., min_length=2, max_length=160)
    tier_class: Literal["parcel", "cold_parcel", "ptl"] = "parcel"
    inner_l_cm: float = Field(..., gt=1.0)
    inner_w_cm: float = Field(..., gt=1.0)
    inner_h_cm: float = Field(..., gt=1.0)
    outer_l_cm: Optional[float] = Field(default=None, gt=1.0)
    outer_w_cm: Optional[float] = Field(default=None, gt=1.0)
    outer_h_cm: Optional[float] = Field(default=None, gt=1.0)
    tare_weight_kg: float = Field(default=0.5, ge=0.0)
    max_payload_kg: float = Field(..., gt=0.1)
    ply_rating: str = "5-Ply"
    burst_strength_kgf_cm2: float = Field(default=12.0, gt=0.0)
    ect_kn_m: float = Field(default=5.2, gt=0.0)
    material_cost_inr: float = Field(..., ge=0.0)
    is_cold_chain: bool = False


class CustomContainerResponse(BaseModel):
    id: int
    container_id: str
    name: str
    tier_class: str
    inner_l_cm: float
    inner_w_cm: float
    inner_h_cm: float
    outer_l_cm: float
    outer_w_cm: float
    outer_h_cm: float
    tare_weight_kg: float
    max_payload_kg: float
    ply_rating: str
    burst_strength_kgf_cm2: float
    ect_kn_m: float
    material_cost_inr: float
    is_cold_chain: bool
    is_active: bool
    created_at: datetime


class ShipmentOutcomeUpdate(BaseModel):
    actual_outcome: Literal["pending", "delivered_intact", "damaged_in_transit"]
    damage_notes: str = ""


class ShipmentRecordResponse(BaseModel):
    id: int
    shipment_index: int
    group_label: str
    recommended_carton_id: str
    box_count: int
    container_spec_summary: str
    transit_mode: str
    estimated_transit_days: str
    net_utilization_pct: float
    padded_utilization_pct: float
    chargeable_weight_kg: float
    direct_cost: float
    expected_damage_cost: float
    damage_risk_probability: float
    co2_emissions_kg: float
    handling_badges: List[str]
    items: List[dict]
    placements: List[dict]
    packing_instructions: str
    actual_outcome: str
    damage_notes: str
    outcome_recorded_at: Optional[datetime] = None


class OrderSummaryResponse(BaseModel):
    order_ref: str
    customer_or_po: str
    source_pin: str
    destination_pin: str
    source_zone: str
    destination_zone: str
    shipping_zone_tier: str
    optimization_priority: str
    total_skus: int
    total_units: int
    total_weight_kg: float
    total_direct_cost: float
    baseline_unoptimized_cost: float
    cost_savings_inr: float
    total_expected_damage_cost: float
    total_effective_landed_cost: float
    total_co2_kg: float
    co2_saved_kg: float
    shipment_count: int
    qa_status_summary: str
    created_at: datetime


class OrderDetailResponse(OrderSummaryResponse):
    shipments: List[ShipmentRecordResponse]


class WarehouseManifestResponse(BaseModel):
    order_ref: str
    customer_or_po: str
    generated_at: datetime
    route_summary: str
    total_direct_cost: float
    total_co2_kg: float
    markdown_manifest: str
    html_manifest: str


class AnalyticsKPIs(BaseModel):
    total_orders: int
    total_shipments: int
    total_spend_inr: float
    total_savings_inr: float
    avg_savings_pct: float
    avg_net_utilization_pct: float
    avg_predicted_risk_pct: float
    total_co2_kg: float
    total_co2_saved_kg: float
    master_sku_count: int
    custom_container_count: int


class ZoneBreakdownItem(BaseModel):
    zone_tier: str
    orders: int
    avg_cost: float
    avg_risk_pct: float


class ContainerUsageItem(BaseModel):
    carton_id: str
    shipments: int
    avg_utilization_pct: float
    total_cost: float


class TrendPointItem(BaseModel):
    order_ref: str
    created_at: str
    direct_cost: float
    savings_inr: float
    avg_risk_pct: float
    co2_kg: float


class QACalibrationSummary(BaseModel):
    evaluated_shipments: int
    intact_count: int
    damaged_count: int
    observed_damage_rate_pct: float
    mean_predicted_rate_pct: float


class AnalyticsSummaryResponse(BaseModel):
    kpis: AnalyticsKPIs
    zone_breakdown: List[ZoneBreakdownItem]
    container_usage: List[ContainerUsageItem]
    recent_trend: List[TrendPointItem]
    qa_calibration: QACalibrationSummary
