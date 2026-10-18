"""Enterprise REST API endpoints for Master SKUs, Custom Boxes, Audits, Manifests, and Analytics."""

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from backend.core.catalog import (
    CONTAINER_CATALOG,
    calculate_item_fragility,
    estimate_item_value_inr,
)
from backend.db.models import CustomContainer, MasterSKU, OrderAudit, ShipmentRecord
from backend.db.session import get_db
from backend.schemas.enterprise import (
    AnalyticsKPIs,
    AnalyticsSummaryResponse,
    ContainerUsageItem,
    CustomContainerCreate,
    CustomContainerResponse,
    MasterSKUCreate,
    MasterSKUResponse,
    MasterSKUUpdate,
    OrderDetailResponse,
    OrderSummaryResponse,
    QACalibrationSummary,
    ShipmentOutcomeUpdate,
    ShipmentRecordResponse,
    TrendPointItem,
    WarehouseManifestResponse,
    ZoneBreakdownItem,
)

enterprise_router = APIRouter()


def _shipment_to_response(rec: ShipmentRecord) -> ShipmentRecordResponse:
    return ShipmentRecordResponse(
        id=rec.id,
        shipment_index=rec.shipment_index,
        group_label=rec.group_label,
        recommended_carton_id=rec.recommended_carton_id,
        box_count=rec.box_count,
        container_spec_summary=rec.container_spec_summary,
        transit_mode=rec.transit_mode,
        estimated_transit_days=rec.estimated_transit_days,
        net_utilization_pct=rec.net_utilization_pct,
        padded_utilization_pct=rec.padded_utilization_pct,
        chargeable_weight_kg=rec.chargeable_weight_kg,
        direct_cost=rec.direct_cost,
        expected_damage_cost=rec.expected_damage_cost,
        damage_risk_probability=rec.damage_risk_probability,
        co2_emissions_kg=rec.co2_emissions_kg,
        handling_badges=json.loads(rec.handling_badges_json or "[]"),
        items=json.loads(rec.items_json or "[]"),
        placements=json.loads(rec.placements_json or "[]"),
        packing_instructions=rec.packing_instructions,
        actual_outcome=rec.actual_outcome,
        damage_notes=rec.damage_notes,
        outcome_recorded_at=rec.outcome_recorded_at,
    )


def _qa_summary(shipments: List[ShipmentRecord]) -> str:
    if not shipments:
        return "Pending"
    outcomes = [s.actual_outcome for s in shipments]
    if any(o == "damaged_in_transit" for o in outcomes):
        damaged = sum(1 for o in outcomes if o == "damaged_in_transit")
        return f"{damaged} Damaged"
    intact = sum(1 for o in outcomes if o == "delivered_intact")
    if intact == len(outcomes):
        return f"{intact}/{len(outcomes)} Intact"
    if intact > 0:
        return f"{intact}/{len(outcomes)} Verified"
    return "Pending"


def _order_to_summary(order: OrderAudit) -> OrderSummaryResponse:
    return OrderSummaryResponse(
        order_ref=order.order_ref,
        customer_or_po=order.customer_or_po,
        source_pin=order.source_pin,
        destination_pin=order.destination_pin,
        source_zone=order.source_zone,
        destination_zone=order.destination_zone,
        shipping_zone_tier=order.shipping_zone_tier,
        optimization_priority=order.optimization_priority,
        total_skus=order.total_skus,
        total_units=order.total_units,
        total_weight_kg=order.total_weight_kg,
        total_direct_cost=order.total_direct_cost,
        baseline_unoptimized_cost=order.baseline_unoptimized_cost,
        cost_savings_inr=order.cost_savings_inr,
        total_expected_damage_cost=order.total_expected_damage_cost,
        total_effective_landed_cost=order.total_effective_landed_cost,
        total_co2_kg=order.total_co2_kg,
        co2_saved_kg=order.co2_saved_kg,
        shipment_count=len(order.shipments),
        qa_status_summary=_qa_summary(order.shipments),
        created_at=order.created_at,
    )


# ------------------------------------------------------------------
# Master SKU CRUD
# ------------------------------------------------------------------
@enterprise_router.get("/skus", response_model=List[MasterSKUResponse])
def list_master_skus(
    search: str = Query(default=""),
    category: str = Query(default=""),
    db: Session = Depends(get_db),
):
    q = db.query(MasterSKU)
    if search.strip():
        term = f"%{search.strip()}%"
        q = q.filter(or_(MasterSKU.sku_code.ilike(term), MasterSKU.name.ilike(term)))
    if category.strip() and category.strip() != "All":
        q = q.filter(MasterSKU.product_category == category.strip())
    return q.order_by(MasterSKU.sku_code.asc()).all()


@enterprise_router.post("/skus", response_model=MasterSKUResponse)
def upsert_master_sku(payload: MasterSKUCreate, db: Session = Depends(get_db)):
    code = payload.sku_code.strip().upper()
    frag = (
        payload.fragility_score
        if payload.fragility_score is not None
        else calculate_item_fragility(
            payload.product_category,
            payload.length_cm,
            payload.width_cm,
            payload.height_cm,
            payload.weight_kg,
        )
    )
    val = estimate_item_value_inr(
        payload.product_category, payload.weight_kg, payload.declared_value_inr
    )

    existing = db.query(MasterSKU).filter(MasterSKU.sku_code == code).first()
    if existing:
        existing.name = payload.name.strip()
        existing.product_category = payload.product_category
        existing.length_cm = payload.length_cm
        existing.width_cm = payload.width_cm
        existing.height_cm = payload.height_cm
        existing.weight_kg = payload.weight_kg
        existing.fragility_score = frag
        existing.declared_value_inr = val
        existing.is_liquid = payload.is_liquid
        existing.orientation_sensitive = payload.orientation_sensitive
        existing.requires_cold_chain = payload.requires_cold_chain
        db.commit()
        db.refresh(existing)
        return existing

    sku = MasterSKU(
        sku_code=code,
        name=payload.name.strip(),
        product_category=payload.product_category,
        length_cm=payload.length_cm,
        width_cm=payload.width_cm,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        fragility_score=frag,
        declared_value_inr=val,
        is_liquid=payload.is_liquid,
        orientation_sensitive=payload.orientation_sensitive,
        requires_cold_chain=payload.requires_cold_chain,
    )
    db.add(sku)
    db.commit()
    db.refresh(sku)
    return sku


@enterprise_router.put("/skus/{sku_code}", response_model=MasterSKUResponse)
def update_master_sku(
    sku_code: str, payload: MasterSKUUpdate, db: Session = Depends(get_db)
):
    sku = db.query(MasterSKU).filter(MasterSKU.sku_code == sku_code.strip().upper()).first()
    if not sku:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_code}' not found")

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        if v is not None:
            setattr(sku, k, v)

    if "fragility_score" not in data or data.get("fragility_score") is None:
        sku.fragility_score = calculate_item_fragility(
            sku.product_category,
            sku.length_cm,
            sku.width_cm,
            sku.height_cm,
            sku.weight_kg,
        )

    db.commit()
    db.refresh(sku)
    return sku


@enterprise_router.delete("/skus/{sku_code}")
def delete_master_sku(sku_code: str, db: Session = Depends(get_db)):
    sku = db.query(MasterSKU).filter(MasterSKU.sku_code == sku_code.strip().upper()).first()
    if not sku:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_code}' not found")
    db.delete(sku)
    db.commit()
    return {"deleted": True, "sku_code": sku.sku_code}


# ------------------------------------------------------------------
# Custom Container CRUD
# ------------------------------------------------------------------
@enterprise_router.get("/containers/custom", response_model=List[CustomContainerResponse])
def list_custom_containers(db: Session = Depends(get_db)):
    return (
        db.query(CustomContainer)
        .filter(CustomContainer.is_active.is_(True))
        .order_by(CustomContainer.created_at.desc())
        .all()
    )


@enterprise_router.post("/containers/custom", response_model=CustomContainerResponse)
def create_custom_container(
    payload: CustomContainerCreate, db: Session = Depends(get_db)
):
    cid = payload.container_id.strip().upper()
    if cid in CONTAINER_CATALOG:
        raise HTTPException(
            status_code=409,
            detail=f"Container ID '{cid}' is reserved by the standard BIS IS 2771 catalog.",
        )

    outer_l = payload.outer_l_cm or round(payload.inner_l_cm + 1.5, 2)
    outer_w = payload.outer_w_cm or round(payload.inner_w_cm + 1.5, 2)
    outer_h = payload.outer_h_cm or round(payload.inner_h_cm + 1.5, 2)

    existing = db.query(CustomContainer).filter(CustomContainer.container_id == cid).first()
    if existing:
        existing.name = payload.name.strip()
        existing.tier_class = payload.tier_class
        existing.inner_l_cm = payload.inner_l_cm
        existing.inner_w_cm = payload.inner_w_cm
        existing.inner_h_cm = payload.inner_h_cm
        existing.outer_l_cm = outer_l
        existing.outer_w_cm = outer_w
        existing.outer_h_cm = outer_h
        existing.tare_weight_kg = payload.tare_weight_kg
        existing.max_payload_kg = payload.max_payload_kg
        existing.ply_rating = payload.ply_rating
        existing.burst_strength_kgf_cm2 = payload.burst_strength_kgf_cm2
        existing.ect_kn_m = payload.ect_kn_m
        existing.material_cost_inr = payload.material_cost_inr
        existing.is_cold_chain = payload.is_cold_chain or payload.tier_class == "cold_parcel"
        existing.is_active = True
        db.commit()
        db.refresh(existing)
        return existing

    box = CustomContainer(
        container_id=cid,
        name=payload.name.strip(),
        tier_class=payload.tier_class,
        inner_l_cm=payload.inner_l_cm,
        inner_w_cm=payload.inner_w_cm,
        inner_h_cm=payload.inner_h_cm,
        outer_l_cm=outer_l,
        outer_w_cm=outer_w,
        outer_h_cm=outer_h,
        tare_weight_kg=payload.tare_weight_kg,
        max_payload_kg=payload.max_payload_kg,
        ply_rating=payload.ply_rating,
        burst_strength_kgf_cm2=payload.burst_strength_kgf_cm2,
        ect_kn_m=payload.ect_kn_m,
        material_cost_inr=payload.material_cost_inr,
        is_cold_chain=payload.is_cold_chain or payload.tier_class == "cold_parcel",
        is_active=True,
    )
    db.add(box)
    db.commit()
    db.refresh(box)
    return box


@enterprise_router.delete("/containers/custom/{container_id}")
def delete_custom_container(container_id: str, db: Session = Depends(get_db)):
    cid = container_id.strip().upper()
    box = db.query(CustomContainer).filter(CustomContainer.container_id == cid).first()
    if not box:
        raise HTTPException(status_code=404, detail=f"Custom container '{cid}' not found")
    db.delete(box)
    db.commit()
    return {"deleted": True, "container_id": cid}


# ------------------------------------------------------------------
# Order History, Warehouse Manifest & QA Feedback
# ------------------------------------------------------------------
@enterprise_router.get("/orders", response_model=List[OrderSummaryResponse])
def list_orders(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    orders = (
        db.query(OrderAudit)
        .options(joinedload(OrderAudit.shipments))
        .order_by(OrderAudit.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_order_to_summary(o) for o in orders]


@enterprise_router.get("/orders/{order_ref}", response_model=OrderDetailResponse)
def get_order_detail(order_ref: str, db: Session = Depends(get_db)):
    order = (
        db.query(OrderAudit)
        .options(joinedload(OrderAudit.shipments))
        .filter(OrderAudit.order_ref == order_ref.strip())
        .first()
    )
    if not order:
        raise HTTPException(status_code=404, detail=f"Order '{order_ref}' not found")

    summary = _order_to_summary(order).model_dump()
    summary["shipments"] = [_shipment_to_response(s) for s in order.shipments]
    return OrderDetailResponse(**summary)


@enterprise_router.get("/orders/{order_ref}/manifest", response_model=WarehouseManifestResponse)
def get_order_manifest(order_ref: str, db: Session = Depends(get_db)):
    order = (
        db.query(OrderAudit)
        .options(joinedload(OrderAudit.shipments))
        .filter(OrderAudit.order_ref == order_ref.strip())
        .first()
    )
    if not order:
        raise HTTPException(status_code=404, detail=f"Order '{order_ref}' not found")

    now = datetime.now(timezone.utc)
    route_summary = (
        f"{order.source_pin} ({order.source_zone}) -> "
        f"{order.destination_pin} ({order.destination_zone}) | {order.shipping_zone_tier}"
    )

    md_lines = [
        f"# WAREHOUSE DISPATCH MANIFEST — {order.order_ref}",
        f"- **Customer / PO:** {order.customer_or_po or 'Ad-Hoc Dispatch'}",
        f"- **Route & SLA Tier:** {route_summary}",
        f"- **Optimization Priority:** {order.optimization_priority}",
        f"- **Total Direct Cost:** Rs. {order.total_direct_cost:,.2f} (Savings vs Oversize: Rs. {order.cost_savings_inr:,.2f})",
        f"- **Total Carbon Footprint:** {order.total_co2_kg:.3f} kg CO2e",
        "",
    ]

    html_sections = []
    for sh in order.shipments:
        badges = json.loads(sh.handling_badges_json or "[]")
        items = json.loads(sh.items_json or "[]")
        placements = json.loads(sh.placements_json or "[]")
        badge_str = ", ".join(badges) if badges else "STANDARD DRY CARGO"

        md_lines.extend(
            [
                f"## Shipment #{sh.shipment_index + 1}: {sh.group_label.upper()}",
                f"- **Container:** {sh.recommended_carton_id} ({sh.container_spec_summary})",
                f"- **Mode & Transit:** {sh.transit_mode} ({sh.estimated_transit_days}) | Chargeable Wt: {sh.chargeable_weight_kg:.1f} kg",
                f"- **Compliance Badges:** {badge_str}",
                f"- **Utilization:** Net {sh.net_utilization_pct:.1f}% / Padded {sh.padded_utilization_pct:.1f}% | Risk: {sh.damage_risk_probability*100:.1f}%",
                "",
                "| SKU | Category | Dimensions (cm) | Wt (kg) | Qty | Fragility |",
                "|---|---|---|---|---|---|",
            ]
        )
        item_rows_html = []
        for it in items:
            md_lines.append(
                f"| {it.get('sku_id')} | {it.get('product_category')} | "
                f"{it.get('length_cm')}x{it.get('width_cm')}x{it.get('height_cm')} | "
                f"{it.get('weight_kg')} | {it.get('quantity', 1)} | {it.get('fragility_score')} |"
            )
            item_rows_html.append(
                f"<tr><td>{html.escape(str(it.get('sku_id')))}</td>"
                f"<td>{html.escape(str(it.get('product_category')))}</td>"
                f"<td>{it.get('length_cm')} x {it.get('width_cm')} x {it.get('height_cm')}</td>"
                f"<td>{it.get('weight_kg')} kg</td>"
                f"<td>{it.get('quantity', 1)}</td></tr>"
            )

        if placements:
            md_lines.extend(["", "**3D Load Coordinates (x, y, z -> dx x dy x dz cm):**"])
            for p in placements:
                md_lines.append(
                    f"- Box {p.get('bin_index', 0) + 1} | `{p.get('sku_id')}` @ "
                    f"({p.get('x_cm')}, {p.get('y_cm')}, {p.get('z_cm')}) size "
                    f"{p.get('dx_cm')}x{p.get('dy_cm')}x{p.get('dz_cm')} cm (pad {p.get('pad_cm')} cm)"
                )

        md_lines.extend(["", "**Packing SOP:**", sh.packing_instructions, ""])

        badge_pills = "".join(
            f"<span style='display:inline-block;padding:3px 8px;margin-right:6px;"
            f"border-radius:4px;background:#e0f2fe;color:#0369a1;font-size:11px;font-weight:700;'>"
            f"{html.escape(b)}</span>"
            for b in (badges or ["STANDARD CARGO"])
        )
        html_sections.append(
            f"""
            <div style="border:1px solid #cbd5e1;border-radius:8px;padding:16px;margin-bottom:18px;">
              <h3 style="margin:0 0 8px 0;">Shipment #{sh.shipment_index + 1} — {html.escape(sh.recommended_carton_id)} ({html.escape(sh.group_label.upper())})</h3>
              <p style="margin:4px 0;font-size:13px;color:#334155;">
                <b>Spec:</b> {html.escape(sh.container_spec_summary)} |
                <b>Mode:</b> {html.escape(sh.transit_mode)} ({html.escape(sh.estimated_transit_days)}) |
                <b>Chargeable Wt:</b> {sh.chargeable_weight_kg:.1f} kg |
                <b>Direct Cost:</b> Rs. {sh.direct_cost:,.2f}
              </p>
              <div style="margin:8px 0 12px 0;">{badge_pills}</div>
              <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:10px;">
                <thead>
                  <tr style="background:#f1f5f9;text-align:left;">
                    <th style="padding:6px;border:1px solid #cbd5e1;">SKU</th>
                    <th style="padding:6px;border:1px solid #cbd5e1;">Category</th>
                    <th style="padding:6px;border:1px solid #cbd5e1;">Dims (cm)</th>
                    <th style="padding:6px;border:1px solid #cbd5e1;">Unit Wt</th>
                    <th style="padding:6px;border:1px solid #cbd5e1;">Qty</th>
                  </tr>
                </thead>
                <tbody>{''.join(item_rows_html)}</tbody>
              </table>
              <div style="background:#f8fafc;padding:10px;border-radius:6px;font-size:12px;white-space:pre-line;">
                <b>Packing SOP:</b><br/>{html.escape(sh.packing_instructions)}
              </div>
            </div>
            """
        )

    html_doc = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Manifest {html.escape(order.order_ref)}</title></head>
<body style="font-family:Inter,Arial,sans-serif;max-width:920px;margin:24px auto;color:#0f172a;">
  <div style="border-bottom:2px solid #0f172a;padding-bottom:12px;margin-bottom:18px;display:flex;justify-content:space-between;">
    <div>
      <h1 style="margin:0;font-size:22px;">PackOptima Enterprise — Dispatch Manifest</h1>
      <p style="margin:4px 0 0 0;font-size:13px;color:#475569;">Order Ref: <b>{html.escape(order.order_ref)}</b> | PO: {html.escape(order.customer_or_po or 'Ad-Hoc')}</p>
    </div>
    <div style="text-align:right;font-size:13px;">
      <div><b>Direct Cost:</b> Rs. {order.total_direct_cost:,.2f}</div>
      <div><b>Savings:</b> Rs. {order.cost_savings_inr:,.2f} | <b>CO2e:</b> {order.total_co2_kg:.2f} kg</div>
    </div>
  </div>
  <p style="font-size:13px;background:#f1f5f9;padding:10px;border-radius:6px;">
    <b>Lane:</b> {html.escape(route_summary)} | <b>Priority:</b> {html.escape(order.optimization_priority)}
  </p>
  {''.join(html_sections)}
  <div style="margin-top:24px;border-top:1px dashed #94a3b8;padding-top:12px;font-size:12px;color:#64748b;">
    Warehouse QA Sign-Off: ________________________ &nbsp;&nbsp;&nbsp; Timestamp (UTC): {now.strftime('%Y-%m-%d %H:%M')}
  </div>
</body>
</html>
"""

    return WarehouseManifestResponse(
        order_ref=order.order_ref,
        customer_or_po=order.customer_or_po,
        generated_at=now,
        route_summary=route_summary,
        total_direct_cost=order.total_direct_cost,
        total_co2_kg=order.total_co2_kg,
        markdown_manifest="\n".join(md_lines),
        html_manifest=html_doc,
    )


@enterprise_router.patch("/shipments/{shipment_id}/outcome", response_model=ShipmentRecordResponse)
def update_shipment_outcome(
    shipment_id: int,
    payload: ShipmentOutcomeUpdate,
    db: Session = Depends(get_db),
):
    rec = db.query(ShipmentRecord).filter(ShipmentRecord.id == shipment_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"Shipment ID {shipment_id} not found")

    rec.actual_outcome = payload.actual_outcome
    rec.damage_notes = payload.damage_notes.strip()
    rec.outcome_recorded_at = (
        datetime.now(timezone.utc) if payload.actual_outcome != "pending" else None
    )
    db.commit()
    db.refresh(rec)
    return _shipment_to_response(rec)


# ------------------------------------------------------------------
# Executive Analytics & ESG Telemetry
# ------------------------------------------------------------------
@enterprise_router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary(db: Session = Depends(get_db)):
    orders = (
        db.query(OrderAudit)
        .options(joinedload(OrderAudit.shipments))
        .order_by(OrderAudit.created_at.asc())
        .all()
    )
    all_shipments = [s for o in orders for s in o.shipments]

    total_orders = len(orders)
    total_shipments = len(all_shipments)
    total_spend = round(sum(o.total_direct_cost for o in orders), 2)
    total_baseline = sum(o.baseline_unoptimized_cost for o in orders)
    total_savings = round(sum(o.cost_savings_inr for o in orders), 2)
    avg_savings_pct = (
        round((total_savings / total_baseline) * 100.0, 2)
        if total_baseline > 0
        else 0.0
    )
    avg_util = (
        round(sum(s.net_utilization_pct for s in all_shipments) / total_shipments, 2)
        if total_shipments > 0
        else 0.0
    )
    avg_risk_pct = (
        round(
            (sum(s.damage_risk_probability for s in all_shipments) / total_shipments)
            * 100.0,
            2,
        )
        if total_shipments > 0
        else 0.0
    )
    total_co2 = round(sum(o.total_co2_kg for o in orders), 3)
    total_co2_saved = round(sum(o.co2_saved_kg for o in orders), 3)
    master_sku_count = db.query(MasterSKU).count()
    custom_box_count = (
        db.query(CustomContainer).filter(CustomContainer.is_active.is_(True)).count()
    )

    kpis = AnalyticsKPIs(
        total_orders=total_orders,
        total_shipments=total_shipments,
        total_spend_inr=total_spend,
        total_savings_inr=total_savings,
        avg_savings_pct=avg_savings_pct,
        avg_net_utilization_pct=avg_util,
        avg_predicted_risk_pct=avg_risk_pct,
        total_co2_kg=total_co2,
        total_co2_saved_kg=total_co2_saved,
        master_sku_count=master_sku_count,
        custom_container_count=custom_box_count,
    )

    # Zone breakdown
    zone_map = defaultdict(list)
    for o in orders:
        zone_map[o.shipping_zone_tier].append(o)

    zone_breakdown = []
    for tier, z_orders in sorted(zone_map.items()):
        z_ships = [s for o in z_orders for s in o.shipments]
        z_avg_cost = round(sum(o.total_direct_cost for o in z_orders) / len(z_orders), 2)
        z_avg_risk = (
            round(
                (sum(s.damage_risk_probability for s in z_ships) / len(z_ships)) * 100.0,
                2,
            )
            if z_ships
            else 0.0
        )
        zone_breakdown.append(
            ZoneBreakdownItem(
                zone_tier=tier,
                orders=len(z_orders),
                avg_cost=z_avg_cost,
                avg_risk_pct=z_avg_risk,
            )
        )

    # Container usage
    cont_map = defaultdict(list)
    for s in all_shipments:
        cont_map[s.recommended_carton_id].append(s)

    container_usage = [
        ContainerUsageItem(
            carton_id=cid,
            shipments=len(s_list),
            avg_utilization_pct=round(
                sum(s.net_utilization_pct for s in s_list) / len(s_list), 2
            ),
            total_cost=round(sum(s.direct_cost for s in s_list), 2),
        )
        for cid, s_list in sorted(
            cont_map.items(), key=lambda kv: len(kv[1]), reverse=True
        )
    ]

    # Recent chronological trend (up to 30)
    recent_trend = []
    for o in orders[-30:]:
        o_risk = (
            round(
                (sum(s.damage_risk_probability for s in o.shipments) / len(o.shipments))
                * 100.0,
                2,
            )
            if o.shipments
            else 0.0
        )
        recent_trend.append(
            TrendPointItem(
                order_ref=o.order_ref,
                created_at=o.created_at.strftime("%Y-%m-%d"),
                direct_cost=o.total_direct_cost,
                savings_inr=o.cost_savings_inr,
                avg_risk_pct=o_risk,
                co2_kg=o.total_co2_kg,
            )
        )

    # QA calibration
    evaluated = [s for s in all_shipments if s.actual_outcome != "pending"]
    intact_cnt = sum(1 for s in evaluated if s.actual_outcome == "delivered_intact")
    damaged_cnt = sum(1 for s in evaluated if s.actual_outcome == "damaged_in_transit")
    obs_rate = (
        round((damaged_cnt / len(evaluated)) * 100.0, 2) if evaluated else 0.0
    )
    pred_rate = (
        round(
            (sum(s.damage_risk_probability for s in evaluated) / len(evaluated))
            * 100.0,
            2,
        )
        if evaluated
        else 0.0
    )

    qa_cal = QACalibrationSummary(
        evaluated_shipments=len(evaluated),
        intact_count=intact_cnt,
        damaged_count=damaged_cnt,
        observed_damage_rate_pct=obs_rate,
        mean_predicted_rate_pct=pred_rate,
    )

    return AnalyticsSummaryResponse(
        kpis=kpis,
        zone_breakdown=zone_breakdown,
        container_usage=container_usage,
        recent_trend=recent_trend,
        qa_calibration=qa_cal,
    )
