"""SQLAlchemy 2.0 ORM models for Packaging Advisor Enterprise."""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all Packaging Advisor database tables."""


class MasterSKU(Base):
    """Enterprise product catalog item stored in relational DB."""

    __tablename__ = "master_skus"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku_code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    product_category: Mapped[str] = mapped_column(String(64), default="General", nullable=False)
    length_cm: Mapped[float] = mapped_column(Float, nullable=False)
    width_cm: Mapped[float] = mapped_column(Float, nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    fragility_score: Mapped[float] = mapped_column(Float, nullable=False)
    declared_value_inr: Mapped[float] = mapped_column(Float, default=1000.0, nullable=False)
    is_liquid: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    orientation_sensitive: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    requires_cold_chain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class CustomContainer(Base):
    """Warehouse-specific custom carton / container registered by ops teams."""

    __tablename__ = "custom_containers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    container_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    tier_class: Mapped[str] = mapped_column(String(32), default="parcel", nullable=False)
    inner_l_cm: Mapped[float] = mapped_column(Float, nullable=False)
    inner_w_cm: Mapped[float] = mapped_column(Float, nullable=False)
    inner_h_cm: Mapped[float] = mapped_column(Float, nullable=False)
    outer_l_cm: Mapped[float] = mapped_column(Float, nullable=False)
    outer_w_cm: Mapped[float] = mapped_column(Float, nullable=False)
    outer_h_cm: Mapped[float] = mapped_column(Float, nullable=False)
    tare_weight_kg: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    max_payload_kg: Mapped[float] = mapped_column(Float, nullable=False)
    ply_rating: Mapped[str] = mapped_column(String(32), default="5-Ply", nullable=False)
    burst_strength_kgf_cm2: Mapped[float] = mapped_column(Float, default=12.0, nullable=False)
    ect_kn_m: Mapped[float] = mapped_column(Float, default=5.2, nullable=False)
    material_cost_inr: Mapped[float] = mapped_column(Float, nullable=False)
    is_cold_chain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class OrderAudit(Base):
    """Persisted optimization audit record per dispatched/evaluated cart."""

    __tablename__ = "order_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_ref: Mapped[str] = mapped_column(String(48), unique=True, index=True, nullable=False)
    customer_or_po: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    source_pin: Mapped[str] = mapped_column(String(12), nullable=False)
    destination_pin: Mapped[str] = mapped_column(String(12), nullable=False)
    source_zone: Mapped[str] = mapped_column(String(32), nullable=False)
    destination_zone: Mapped[str] = mapped_column(String(32), nullable=False)
    shipping_zone_tier: Mapped[str] = mapped_column(String(80), nullable=False)
    optimization_priority: Mapped[str] = mapped_column(String(32), default="balanced", nullable=False)
    total_skus: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_units: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_weight_kg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_direct_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    baseline_unoptimized_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_savings_inr: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_expected_damage_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_effective_landed_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_co2_kg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    co2_saved_kg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    shipments: Mapped[List["ShipmentRecord"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
        order_by="ShipmentRecord.shipment_index",
    )


class ShipmentRecord(Base):
    """Individual compatible shipment segment linked to an OrderAudit."""

    __tablename__ = "shipment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("order_audits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    shipment_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    group_label: Mapped[str] = mapped_column(String(48), default="general", nullable=False)
    recommended_carton_id: Mapped[str] = mapped_column(String(64), nullable=False)
    box_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    container_spec_summary: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    transit_mode: Mapped[str] = mapped_column(String(48), default="Surface", nullable=False)
    estimated_transit_days: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    net_utilization_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    padded_utilization_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    chargeable_weight_kg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    direct_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    expected_damage_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    damage_risk_probability: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    co2_emissions_kg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    handling_badges_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    items_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    placements_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    packing_instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    actual_outcome: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    damage_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    outcome_recorded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    order: Mapped["OrderAudit"] = relationship(back_populates="shipments")
