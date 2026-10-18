from datetime import datetime, timezone
import json
from typing import List
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.catalog import (
    CONTAINER_CATALOG,
    ORDERED_CONTAINERS,
    PRODUCT_CATEGORIES,
    ContainerSpec,
    compute_cushion_pad_cm,
)
from backend.db.models import CustomContainer, OrderAudit, ShipmentRecord
from backend.db.session import get_db
from backend.genai.service import ExplainerService
from backend.ml.predictor import get_damage_predictor
from backend.optimization.solver import PackingSolver
from backend.schemas.cart import (
    BoxAlternative,
    CartRequest,
    ItemPlacement,
    RecommendationResponse,
    ShipmentResponse,
)
from backend.services.cost_engine import DynamicCostEngine
from backend.services.grouping import group_cart_items
from backend.services.routing_engine import RouteProfile, RoutingEngine

api_router = APIRouter()

# Module-level singletons (avoid per-request disk deserialization)
_routing_engine = RoutingEngine()
_cost_engine = DynamicCostEngine()
_cost_matrix = _cost_engine.get_current_rates()
_packing_solver = PackingSolver()
_damage_predictor = get_damage_predictor()
_explainer_service = ExplainerService()


def _load_custom_containers(db: Session) -> List[ContainerSpec]:
    rows = (
        db.query(CustomContainer)
        .filter(CustomContainer.is_active.is_(True))
        .order_by(CustomContainer.id.asc())
        .all()
    )
    specs: List[ContainerSpec] = []
    for r in rows:
        specs.append(
            ContainerSpec(
                container_id=r.container_id,
                name=r.name,
                tier_class=r.tier_class,
                inner_dims_cm=(r.inner_l_cm, r.inner_w_cm, r.inner_h_cm),
                outer_dims_cm=(r.outer_l_cm, r.outer_w_cm, r.outer_h_cm),
                tare_weight_kg=r.tare_weight_kg,
                max_payload_kg=r.max_payload_kg,
                ply_rating=r.ply_rating,
                burst_strength_kgf_cm2=r.burst_strength_kgf_cm2,
                ect_kn_m=r.ect_kn_m,
                material_cost_inr=r.material_cost_inr,
                is_cold_chain=r.is_cold_chain,
                shipping_flat=False,
                flat_lane_rate_inr=0.0,
            )
        )
    return specs


@api_router.get("/health")
def health_check(db: Session = Depends(get_db)):
    custom_count = db.query(CustomContainer).filter(CustomContainer.is_active.is_(True)).count()
    return {
        "status": "healthy",
        "ml_model_loaded": _damage_predictor.model_loaded,
        "database_connected": True,
        "custom_containers": custom_count,
        **_explainer_service.get_provider_status(),
    }


@api_router.get("/catalog")
def get_catalog(db: Session = Depends(get_db)):
    """Return canonical product categories and standard + custom containers."""
    categories = [
        {
            "code": spec.code,
            "name": spec.name,
            "fragility_range": list(spec.fragility_range),
            "base_fragility": spec.base_fragility,
            "default_value_per_kg": spec.default_value_per_kg,
            "esd_sensitive": spec.esd_sensitive,
            "requires_cold_chain": spec.requires_cold_chain,
            "heavy_duty": spec.heavy_duty,
        }
        for spec in PRODUCT_CATEGORIES.values()
    ]
    all_containers = list(ORDERED_CONTAINERS) + _load_custom_containers(db)
    standard_ids = set(CONTAINER_CATALOG.keys())
    containers = [
        {
            "container_id": c.container_id,
            "name": c.name,
            "tier_class": c.tier_class,
            "inner_dims_cm": list(c.inner_dims_cm),
            "outer_dims_cm": list(c.outer_dims_cm),
            "tare_weight_kg": c.tare_weight_kg,
            "max_payload_kg": c.max_payload_kg,
            "ply_rating": c.ply_rating,
            "burst_strength_kgf_cm2": c.burst_strength_kgf_cm2,
            "ect_kn_m": c.ect_kn_m,
            "material_cost_inr": c.material_cost_inr,
            "is_cold_chain": c.is_cold_chain,
            "shipping_flat": c.shipping_flat,
            "flat_lane_rate_inr": c.flat_lane_rate_inr,
            "is_custom": c.container_id not in standard_ids,
        }
        for c in all_containers
    ]
    return {"categories": categories, "containers": containers}


def _compute_handling_badges(
    group_label: str,
    group: list,
    highest_frag: float,
    gross_wt: float,
) -> List[str]:
    badges: List[str] = []
    categories = {i.product_category for i in group}
    if group_label == "perishables" or any(
        getattr(i, "requires_cold_chain", False) for i in group
    ):
        badges.append("COLD CHAIN 2-8 C")
    if any(i.is_liquid for i in group):
        badges.append("LIQUID CONTAINMENT")
    if any(i.orientation_sensitive for i in group):
        badges.append("THIS WAY UP")
    if "Precision Electronics" in categories:
        badges.append("ESD PROTECTED")
    if highest_frag >= 0.70:
        badges.append("ISTA 3A CUSHIONED")
    if gross_wt >= 15.0:
        badges.append("HEAVY LIFT (>15 KG)")
    return badges


def _estimate_unoptimized_baseline(
    group: list,
    best: dict,
    route: RouteProfile,
) -> tuple[float, float]:
    """Simulate realistic unoptimized warehouse packing (+2 box sizes in tier)."""
    box_id = best["box_id"]
    best_spec = CONTAINER_CATALOG.get(box_id)
    if (
        best_spec is None
        or best_spec.tier_class in ("ptl", "reefer_ptl", "ftl")
        or "MULTI-TRUCK" in box_id
    ):
        return round(best["t_cost"] * 1.18, 2), round(best["co2_kg"] * 1.15, 3)

    tier_specs = [
        s for s in ORDERED_CONTAINERS if s.tier_class == best_spec.tier_class
    ]
    ids = [s.container_id for s in tier_specs]
    if box_id not in ids:
        return round(best["t_cost"] * 1.18, 2), round(best["co2_kg"] * 1.15, 3)

    idx = ids.index(box_id)
    if idx >= len(tier_specs) - 1:
        return round(best["t_cost"] * 1.18, 2), round(best["co2_kg"] * 1.15, 3)

    target_idx = min(len(tier_specs) - 1, idx + 2)
    target_spec = tier_specs[target_idx]
    if best_spec.tier_class == "parcel" and best_spec.inner_volume_cm3 < 30_000.0:
        heavy_idx = ids.index("C-HEAVY-5P")
        target_spec = tier_specs[min(target_idx, heavy_idx)]

    box_count = max(1, int(best.get("box_count", 1)))
    cand = {
        "box_id": target_spec.container_id,
        "box_count": box_count,
        "b_cost": target_spec.material_cost_inr * box_count,
        "box_vol": target_spec.inner_volume_cm3 * box_count,
        "outer_vol": target_spec.outer_volume_cm3 * box_count,
        "tare_wt": target_spec.tare_weight_kg * box_count,
        "utilization_pct": min(
            0.95,
            best["util_pct"]
            * (best_spec.inner_volume_cm3 / max(1.0, target_spec.inner_volume_cm3)),
        ),
        "padded_utilization_pct": min(
            0.98,
            best["padded_util_pct"]
            * (best_spec.inner_volume_cm3 / max(1.0, target_spec.inner_volume_cm3)),
        ),
        "spec": target_spec,
    }
    modes = _cost_engine.get_eligible_modes(cand, route)
    matching = [m for m in modes if m[0] == best["mode_name"]]
    mode_name, base_rate, _, _ = matching[0] if matching else modes[0]
    pricing = _cost_engine.calculate_mode_pricing(group, cand, route, mode_name, base_rate)
    return max(best["t_cost"], pricing["direct_cost"]), max(best["co2_kg"], pricing["co2_kg"])


def _build_packing_instructions(
    group: list,
    spec_summary: str,
    box_count: int,
    max_pad_cm: float,
    highest_frag: float,
    categories: set,
) -> str:
    instructions = [
        "**Moisture Barrier (Monsoon Protection):** Indian transit faces high "
        "humidity that can cut corrugated ECT strength by up to 40%. Wrap inner "
        "cargo in sealed LDPE shrink-wrap before placing into the outer box.",
        f"**Tamper-Evident Sealing ({spec_summary or 'IS 2771 Box'}):** Apply H-taping "
        "along all seams using cross-strapped BOPP security tape across each carton"
        + (f" ({box_count} cartons total)." if box_count > 1 else "."),
    ]

    if max_pad_cm > 0:
        instructions.append(
            f"**ISTA 3A Cushioning Clearance ({max_pad_cm:.1f} cm/face):** Surround "
            "delicate surfaces with EPE foam planks or honeycomb kraft pads matching "
            "the allocated clearance buffer so corners never bottom out."
        )
    if (
        "Fine Art & Antiques" in categories
        or "Glass & Ceramics" in categories
        or highest_frag >= 0.80
    ):
        instructions.append(
            "**High-Impact Suspension:** Double-box or insert corner edge-protectors "
            "rather than loose bubble wrap for glass, ceramics, and antiques."
        )
    if "Precision Electronics" in categories:
        instructions.append(
            "**Heat & ESD Defence:** Seal components inside metallized ESD bags "
            "with Silica Gel desiccant sachets to prevent condensation (+40 C transit)."
        )
    if "Perishable Goods" in categories or any(
        getattr(i, "requires_cold_chain", False) for i in group
    ):
        instructions.append(
            "**Thermal Cold-Chain Protocol:** Pre-chill insulated EPS/PCM gel packs "
            "to target temperature and pack tightly against perishables with a liner barrier."
        )
    if any(i.is_liquid for i in group):
        instructions.append(
            "**Leak Containment:** Tape-lock bottle caps, enclose in primary sealed "
            "ziplock pouches, and line bottom with absorbent cellulose wadding."
        )
    if any(i.orientation_sensitive for i in group):
        instructions.append(
            "**Orientation Lock:** Affix high-contrast 'THIS WAY UP' and fragile "
            "arrows on all four vertical faces."
        )
    if highest_frag < 0.30:
        instructions.append(
            "**Void Fill:** Pack remaining pockets tightly with crushed kraft paper "
            "to stop load shifting during road vibration."
        )
    else:
        instructions.append(
            "**Immobilisation:** Eliminate internal rattle by blocking all remaining "
            "void gaps with dense EPE foam blocks or inflatable air pillows."
        )

    return "\n\n".join(instructions)


def _select_top_alternatives(all_options: list) -> list[BoxAlternative]:
    selected = []
    seen_keys = set()

    def _add(opt: dict, label_prefix: str):
        key = (opt["box_id"], opt["box_count"], opt["mode_name"])
        if key in seen_keys or len(selected) >= 3:
            return
        seen_keys.add(key)
        carton_tag = (
            f"{opt['box_count']}x {opt['box_id']}"
            if opt["box_count"] > 1
            else opt["box_id"]
        )
        selected.append(
            BoxAlternative(
                tier_name=f"{label_prefix} ({opt['mode_name']})",
                carton_id=carton_tag,
                total_cost=opt["t_cost"],
                utilization_pct=opt["util_pct"],
                risk_probability=opt["r_prob"],
                cost_breakdown=opt["cost_breakdown"],
                box_count=opt["box_count"],
                transit_mode=opt["mode_name"],
                estimated_days=opt["est_days"],
                expected_damage_cost=opt["expected_damage_cost"],
                effective_landed_cost=opt["effective_landed_cost"],
                board_spec=opt["spec_summary"],
                co2_kg=opt["co2_kg"],
            )
        )

    if all_options:
        _add(all_options[0], "Best Fit")
        cheapest_cost = min(all_options, key=lambda x: (x["t_cost"], x["r_prob"]))
        _add(cheapest_cost, "Lowest Freight")
        safest = min(all_options, key=lambda x: (x["r_prob"], x["t_cost"]))
        _add(safest, "Lowest Risk")
        for opt in all_options[1:]:
            if len(selected) >= 3:
                break
            _add(opt, "Alternative")

    return selected


@api_router.post("/recommend", response_model=RecommendationResponse)
def recommend_packaging(request: CartRequest, db: Session = Depends(get_db)):
    route = _routing_engine.analyze_route(
        request.source_pin, request.destination_pin
    )
    groups, grouping_reasons = group_cart_items(request.items)

    if not groups:
        return RecommendationResponse(
            shipments=[],
            total_cart_cost=0.0,
            source_zone=route.source_zone,
            destination_zone=route.destination_zone,
            shipping_zone_tier=route.shipping_zone_tier,
        )

    custom_containers = _load_custom_containers(db)

    shipments: List[ShipmentResponse] = []
    shipment_persist_payloads: List[dict] = []
    total_cart_cost = 0.0
    baseline_unoptimized_cost = 0.0
    total_expected_damage_cost = 0.0
    total_effective_landed_cost = 0.0
    total_co2_kg = 0.0
    baseline_total_co2_kg = 0.0

    for idx_grp, (group_label, group) in enumerate(groups):
        require_cold_chain = group_label == "perishables"
        solver_results = _packing_solver.solve_3d_bin_packing(
            group,
            _cost_matrix,
            require_cold_chain=require_cold_chain,
            extra_containers=custom_containers,
        )

        highest_frag = max(i.fragility_score for i in group)
        max_pad_cm = max(
            compute_cushion_pad_cm(i.fragility_score, i.weight_kg) for i in group
        )
        categories = {i.product_category for i in group}

        all_options = []
        for cand in solver_results:
            spec = cand.get("spec")
            spec_summary = (
                f"{spec.ply_rating}, BS {spec.burst_strength_kgf_cm2:.0f} kgf/cm2, "
                f"ECT {spec.ect_kn_m:.1f} kN/m"
                if spec is not None
                else "Custom / Multi-Truck"
            )
            modes = _cost_engine.get_eligible_modes(cand, route)

            for mode_name, base_rate, mode_advice, est_days in modes:
                transit_days_min = (
                    route.air_days_min
                    if mode_name == "Air"
                    else route.surface_days_min
                )
                r_prob = _damage_predictor.predict_damage_risk(
                    group,
                    cand["utilization_pct"],
                    route.distance_multiplier,
                    padded_utilization_pct=cand["padded_utilization_pct"],
                    cushion_adequacy_ratio=cand["cushion_adequacy_ratio"],
                    container_spec=spec,
                    transit_mode=mode_name,
                    handling_nodes=route.handling_nodes,
                    transit_days_min=transit_days_min,
                    box_count=cand["box_count"],
                )
                pricing = _cost_engine.calculate_mode_pricing(
                    group, cand, route, mode_name, base_rate
                )
                expected_damage_cost = round(
                    r_prob * pricing["declared_value_inr"] * 0.30, 2
                )
                effective_landed_cost = round(
                    pricing["direct_cost"] + expected_damage_cost, 2
                )

                box_id = cand["box_id"]
                is_ftl = any(tag in box_id for tag in ("TRUCK", "MULTI"))
                is_ptl = any(tag in box_id for tag in ("PALLET", "REEFER"))
                if is_ftl:
                    cost_label = "Loading & Crating"
                elif is_ptl:
                    cost_label = "Pallet & Handling"
                else:
                    cost_label = "Container"

                cb_parts = [f"{cost_label}: Rs.{pricing['material_cost']:,.0f}"]
                if pricing["dunnage_cost"] > 0:
                    cb_parts.append(f"Dunnage: Rs.{pricing['dunnage_cost']:,.0f}")
                if (spec is not None and spec.shipping_flat) or "MULTI-TRUCK" in box_id:
                    cb_parts.append(f"Freight: Rs.{pricing['freight_cost']:,.0f}")
                else:
                    cb_parts.append(
                        f"Chargeable Wt: {pricing['chargeable_wt_kg']:.1f} kg "
                        f"(Vol: {pricing['vol_wt_kg']:.1f} kg)"
                    )
                    cb_parts.append(f"Freight: Rs.{pricing['freight_cost']:,.0f}")

                all_options.append(
                    {
                        "box_id": box_id,
                        "box_count": cand["box_count"],
                        "mode_name": mode_name,
                        "mode_advice": mode_advice,
                        "est_days": est_days,
                        "t_cost": pricing["direct_cost"],
                        "m_cost": pricing["material_cost"],
                        "dunnage_cost": pricing["dunnage_cost"],
                        "s_cost": pricing["freight_cost"],
                        "r_prob": r_prob,
                        "util_pct": cand["utilization_pct"],
                        "padded_util_pct": cand["padded_utilization_pct"],
                        "cushion_ratio": cand["cushion_adequacy_ratio"],
                        "charge_wt": pricing["chargeable_wt_kg"],
                        "gross_wt": pricing["gross_wt_kg"],
                        "vol_wt": pricing["vol_wt_kg"],
                        "declared_value_inr": pricing["declared_value_inr"],
                        "expected_damage_cost": expected_damage_cost,
                        "effective_landed_cost": effective_landed_cost,
                        "co2_kg": pricing["co2_kg"],
                        "container_dims_cm": cand["container_dims_cm"],
                        "spec_summary": spec_summary,
                        "placements": cand["placements"],
                        "cost_breakdown": " | ".join(cb_parts),
                    }
                )

        all_options = _cost_engine.sort_options(
            all_options, request.optimization_priority
        )
        best = all_options[0]
        alternatives = _select_top_alternatives(all_options)

        base_cost, base_co2 = _estimate_unoptimized_baseline(group, best, route)
        baseline_unoptimized_cost += base_cost
        baseline_total_co2_kg += base_co2

        total_cart_cost += best["t_cost"]
        total_expected_damage_cost += best["expected_damage_cost"]
        total_effective_landed_cost += best["effective_landed_cost"]
        total_co2_kg += best["co2_kg"]

        explanation, gen_instructions = _explainer_service.generate_summary(
            best["box_id"],
            best["r_prob"],
            best["t_cost"],
            best["util_pct"],
            group,
            request.use_genai,
            request.model_provider,
            request.model_name,
            box_count=best["box_count"],
            spec_summary=best["spec_summary"],
            zone_tier=route.shipping_zone_tier,
            transit_mode=best["mode_name"],
            transit_days=best["est_days"],
            chargeable_wt_kg=best["charge_wt"],
            volumetric_wt_kg=best["vol_wt"],
            dunnage_cost=best["dunnage_cost"],
            padded_utilization_pct=best["padded_util_pct"],
        )

        best_label = (
            f"{best['box_count']}x {best['box_id']}"
            if best["box_count"] > 1
            else best["box_id"]
        )
        carton_reasoning = (
            f"Selected {best_label} ({best['spec_summary']}) via {best['mode_name']} "
            f"[{best['est_days']}] under '{request.optimization_priority}' priority -- "
            f"direct cost Rs.{best['t_cost']:,.0f}, CO2e {best['co2_kg']:.2f} kg."
        )
        cost_breakdown = (
            f"Container: Rs.{best['m_cost']:,.0f}\n"
            f"Dunnage: Rs.{best['dunnage_cost']:,.0f}\n"
            f"Freight ({best['charge_wt']:.1f} kg billable): Rs.{best['s_cost']:,.0f}\n"
            f"Total Direct: Rs.{best['t_cost']:,.0f}"
        )
        risk_reasoning = (
            f"Calibrated damage probability {best['r_prob']*100:.1f}% "
            f"(net util {best['util_pct']*100:.1f}%, padded util {best['padded_util_pct']*100:.1f}%, "
            f"cushion adequacy {best['cushion_ratio']*100:.0f}%, {route.handling_nodes} handling nodes)."
        )
        packing_instructions = gen_instructions or _build_packing_instructions(
            group,
            best["spec_summary"],
            best["box_count"],
            max_pad_cm,
            highest_frag,
            categories,
        )
        handling_badges = _compute_handling_badges(
            group_label, group, highest_frag, best["gross_wt"]
        )

        shipments.append(
            ShipmentResponse(
                items=group,
                recommended_carton_id=best_label,
                total_estimated_cost=best["t_cost"],
                damage_risk_probability=best["r_prob"],
                genai_explanation=explanation,
                carton_reasoning=carton_reasoning,
                cost_breakdown=cost_breakdown,
                risk_reasoning=risk_reasoning,
                packing_instructions=packing_instructions,
                transit_mode_advice=best["mode_advice"],
                alternatives=alternatives,
                group_label=group_label,
                box_count=best["box_count"],
                container_inner_dims_cm=best["container_dims_cm"],
                container_spec_summary=best["spec_summary"],
                net_utilization_pct=round(best["util_pct"] * 100.0, 2),
                padded_utilization_pct=round(best["padded_util_pct"] * 100.0, 2),
                chargeable_weight_kg=best["charge_wt"],
                actual_gross_weight_kg=best["gross_wt"],
                volumetric_weight_kg=best["vol_wt"],
                declared_value_inr=best["declared_value_inr"],
                expected_damage_cost=best["expected_damage_cost"],
                estimated_transit_days=best["est_days"],
                placements=[ItemPlacement(**p) for p in best["placements"]],
                transit_mode=best["mode_name"],
                co2_emissions_kg=best["co2_kg"],
                handling_badges=handling_badges,
            )
        )

        shipment_persist_payloads.append(
            {
                "shipment_index": idx_grp,
                "group_label": group_label,
                "recommended_carton_id": best_label,
                "box_count": best["box_count"],
                "container_spec_summary": best["spec_summary"],
                "transit_mode": best["mode_name"],
                "estimated_transit_days": best["est_days"],
                "net_utilization_pct": round(best["util_pct"] * 100.0, 2),
                "padded_utilization_pct": round(best["padded_util_pct"] * 100.0, 2),
                "chargeable_weight_kg": best["charge_wt"],
                "direct_cost": best["t_cost"],
                "expected_damage_cost": best["expected_damage_cost"],
                "damage_risk_probability": best["r_prob"],
                "co2_emissions_kg": best["co2_kg"],
                "handling_badges_json": json.dumps(handling_badges),
                "items_json": json.dumps([i.model_dump() for i in group]),
                "placements_json": json.dumps(best["placements"]),
                "packing_instructions": packing_instructions,
            }
        )

    cost_savings_inr = round(max(0.0, baseline_unoptimized_cost - total_cart_cost), 2)
    cost_savings_pct = (
        round((cost_savings_inr / baseline_unoptimized_cost) * 100.0, 2)
        if baseline_unoptimized_cost > 0
        else 0.0
    )
    co2_saved_kg = round(max(0.0, baseline_total_co2_kg - total_co2_kg), 3)

    order_ref = ""
    if request.persist_history:
        order_ref = (
            f"ORD-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        )
        total_units = sum(i.quantity for i in request.items)
        total_weight_kg = round(
            sum(i.weight_kg * i.quantity for i in request.items), 2
        )
        audit = OrderAudit(
            order_ref=order_ref,
            customer_or_po=(request.customer_or_po or "").strip() or "Ad-Hoc Dispatch",
            source_pin=route.source_pin,
            destination_pin=route.destination_pin,
            source_zone=route.source_zone,
            destination_zone=route.destination_zone,
            shipping_zone_tier=route.shipping_zone_tier,
            optimization_priority=request.optimization_priority,
            total_skus=len(request.items),
            total_units=total_units,
            total_weight_kg=total_weight_kg,
            total_direct_cost=round(total_cart_cost, 2),
            baseline_unoptimized_cost=round(baseline_unoptimized_cost, 2),
            cost_savings_inr=cost_savings_inr,
            total_expected_damage_cost=round(total_expected_damage_cost, 2),
            total_effective_landed_cost=round(total_effective_landed_cost, 2),
            total_co2_kg=round(total_co2_kg, 3),
            co2_saved_kg=co2_saved_kg,
        )
        db.add(audit)
        db.flush()

        created_records: List[ShipmentRecord] = []
        for pld in shipment_persist_payloads:
            rec = ShipmentRecord(order_id=audit.id, **pld)
            db.add(rec)
            created_records.append(rec)

        db.commit()
        for ship_resp, rec in zip(shipments, created_records):
            ship_resp.shipment_db_id = rec.id

    compatibility_warning = ""
    if len(groups) > 1:
        compatibility_warning = (
            f"Smart Grouping split your cart into {len(groups)} safe shipments: "
            + " ".join(grouping_reasons)
        )

    return RecommendationResponse(
        shipments=shipments,
        total_cart_cost=round(total_cart_cost, 2),
        compatibility_warning=compatibility_warning,
        source_zone=route.source_zone,
        destination_zone=route.destination_zone,
        shipping_zone_tier=route.shipping_zone_tier,
        total_expected_damage_cost=round(total_expected_damage_cost, 2),
        total_effective_landed_cost=round(total_effective_landed_cost, 2),
        grouping_reasons=grouping_reasons,
        order_ref=order_ref,
        baseline_unoptimized_cost=round(baseline_unoptimized_cost, 2),
        cost_savings_inr=cost_savings_inr,
        cost_savings_pct=cost_savings_pct,
        total_co2_kg=round(total_co2_kg, 3),
        co2_saved_kg=co2_saved_kg,
    )
