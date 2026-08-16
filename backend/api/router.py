from fastapi import APIRouter
from backend.schemas.cart import CartRequest, RecommendationResponse
from backend.ml.predictor import DamagePredictor
from backend.optimization.solver import PackingSolver
from backend.services.cost_engine import DynamicCostEngine
from backend.genai.service import ExplainerService

api_router = APIRouter()

@api_router.get("/health")
def health_check():
    return {"status": "healthy"}

@api_router.post("/recommend", response_model=RecommendationResponse)
def recommend_packaging(request: CartRequest):
    # ------------------------------------------------------------------
    # 1. Smart Grouping — separate incompatible cargo types
    # ------------------------------------------------------------------
    # NOTE: Perishable items that are also liquid stay in the perishables
    # group (caught first).  The liquids group only captures non-perishable
    # liquids (e.g. industrial chemicals, beverages without cold-chain need).
    perishables = []
    liquids = []
    electronics = []
    general = []

    for item in request.items:
        if item.product_category == "Perishable Goods":
            perishables.append(item)
        elif item.is_liquid:
            liquids.append(item)
        elif item.product_category == "Precision Electronics":
            electronics.append(item)
        else:
            general.append(item)

    groups = []
    if perishables:
        groups.append(("perishables", perishables))
    if liquids:
        groups.append(("liquids", liquids))
    if electronics:
        groups.append(("electronics", electronics))
    if general:
        groups.append(("general", general))

    if not groups:
        return RecommendationResponse(shipments=[], total_cart_cost=0.0)

    # ------------------------------------------------------------------
    # 2. Shared services
    # ------------------------------------------------------------------
    cost_engine = DynamicCostEngine()
    cost_matrix = cost_engine.get_current_rates()
    predictor = DamagePredictor()
    explainer = ExplainerService()
    solver = PackingSolver()

    src = str(request.source_pin).strip() if request.source_pin else "110001"
    dst = str(request.destination_pin).strip() if request.destination_pin else "560001"

    if len(src) < 2 or not src.isdigit():
        src = "110001"
    if len(dst) < 2 or not dst.isdigit():
        dst = "560001"

    def get_zone(pin: str) -> str:
        first_digit = pin[0]
        if first_digit in ['1', '2']:
            return "North"
        elif first_digit in ['3', '4']:
            return "West"
        elif first_digit in ['5', '6']:
            return "South"
        elif first_digit in ['7', '8']:
            return "East"
        elif first_digit == '9':
            return "APS"
        return "Unknown"

    source_zone = get_zone(src)
    dest_zone = get_zone(dst)

    # Zone-to-zone distance multiplier for pricing
    dist_matrix = {
        "North":   {"North": 1.2, "South": 2.5, "East": 2.0, "West": 1.8, "APS": 3.0, "Unknown": 2.0},
        "South":   {"North": 2.5, "South": 1.2, "East": 1.8, "West": 1.8, "APS": 3.0, "Unknown": 2.0},
        "East":    {"North": 2.0, "South": 1.8, "East": 1.2, "West": 2.5, "APS": 3.0, "Unknown": 2.0},
        "West":    {"North": 1.8, "South": 1.8, "East": 2.5, "West": 1.2, "APS": 3.0, "Unknown": 2.0},
        "APS":     {"North": 3.0, "South": 3.0, "East": 3.0, "West": 3.0, "APS": 1.2, "Unknown": 3.0},
        "Unknown": {"North": 2.0, "South": 2.0, "East": 2.0, "West": 2.0, "APS": 3.0, "Unknown": 2.0},
    }

    is_same_subzone = (src[:2] == dst[:2])

    if is_same_subzone:
        distance_multiplier = 1.0
    else:
        distance_multiplier = dist_matrix.get(source_zone, {}).get(dest_zone, 2.0)

    # ------------------------------------------------------------------
    # 3. Process each group
    # ------------------------------------------------------------------
    from backend.schemas.cart import BoxAlternative, ShipmentResponse

    shipments = []
    total_cart_cost = 0.0

    for group_label, group in groups:
        solver_results = solver.solve_3d_bin_packing(group, cost_matrix)

        # For perishable groups, prefer reefer-capable containers
        if group_label == "perishables":
            reefer_results = [r for r in solver_results if "REEFER" in r["box_id"]]
            if reefer_results:
                solver_results = reefer_results

        highest_frag = max(i.fragility_score for i in group)
        categories = set(i.product_category for i in group)
        group_wt = sum(i.weight_kg for i in group)

        # -- Valid transit modes for this group --------------------------
        valid_modes = []
        if is_same_subzone:
            valid_modes.append(("Last-Mile", 25.0, "Last-Mile Road Delivery -- hyper-local transit."))
        elif "Perishable Goods" in categories:
            valid_modes.append(("Refrigerated Road", 20.0,
                                "Cold-Chain Transport -- temperature-controlled surface transit."))
            valid_modes.append(("Air", 60.0,
                                "Express Air Freight -- speed prioritised for perishables."))
        else:
            valid_modes.append(("Surface", 15.0,
                                f"Surface Freight -- cost-effective for {source_zone} to {dest_zone}."))
            valid_modes.append(("Air", 60.0,
                                f"Standard Air Freight -- fast transit for {source_zone} to {dest_zone}."))

        # -- Build all (mode x box) combinations -------------------------
        all_options = []

        for mode_name, base_rate, mode_advice in valid_modes:
            for idx, res in enumerate(solver_results):
                box_id = res["box_id"]
                util_pct = res["utilization_pct"]
                box_vol = res.get("box_vol", 0)

                r_prob = predictor.predict_damage_risk(group, util_pct)

                vol_wt = box_vol / 5000.0
                charge_wt = max(group_wt, vol_wt)

                # Look up cost info
                b_costs = cost_matrix.get(box_id)
                if not b_costs:
                    if "MULTI-TRUCK" in box_id:
                        # Parse truck count from id like "MULTI-TRUCK(2x40ft)"
                        try:
                            trucks = int(box_id.split("(")[1].split("x")[0])
                        except (IndexError, ValueError):
                            trucks = 2
                        b_costs = {
                            "material": trucks * 8000.0,
                            "shipping_flat": True,
                            "flat_rate": trucks * 28000.0,
                        }
                    else:
                        b_costs = {"material": min((box_vol / 1000.0) * 22.5, 5000.0)}

                m_cost = b_costs.get("material", 0.0)

                if b_costs.get("shipping_flat"):
                    s_cost = b_costs.get("flat_rate", 50000.0) * distance_multiplier
                else:
                    s_cost = charge_wt * base_rate * distance_multiplier

                t_cost = m_cost + s_cost

                if idx == 0:
                    box_tier = "Best Fit"
                elif idx == 1:
                    box_tier = "Safest Option"
                else:
                    box_tier = "Oversized"

                tier = f"{box_tier} ({mode_name})"

                # Contextual label for the fixed cost component
                is_ftl = any(tag in box_id for tag in ["TRUCK", "MULTI"])
                is_ptl = any(tag in box_id for tag in ["PALLET", "REEFER"])
                if is_ftl:
                    cost_label = "Loading & Handling"
                elif is_ptl:
                    cost_label = "Pallet & Handling"
                else:
                    cost_label = "Packaging"

                if b_costs.get("shipping_flat"):
                    cb = (f"{cost_label}: Rs.{m_cost:,.0f} | "
                          f"Freight: Rs.{s_cost:,.0f}")
                else:
                    cb = (f"{cost_label}: Rs.{m_cost:,.0f} | "
                          f"Chargeable Wt: {charge_wt:.1f} kg "
                          f"(Vol wt: {vol_wt:.1f} kg) | "
                          f"Shipping: Rs.{s_cost:,.0f}")

                all_options.append({
                    "box_id": box_id,
                    "mode_name": mode_name,
                    "mode_advice": mode_advice,
                    "t_cost": t_cost,
                    "m_cost": m_cost,
                    "s_cost": s_cost,
                    "r_prob": r_prob,
                    "util_pct": util_pct,
                    "charge_wt": charge_wt,
                    "vol_wt": vol_wt,
                    "tier_name": tier,
                    "cost_breakdown": cb,
                })

        # -- Pick best + alternatives -----------------------------------
        all_options.sort(key=lambda x: x["t_cost"])
        best_option = all_options[0]

        alternatives = []
        seen_tiers = set()
        for opt in all_options:
            if opt["tier_name"] not in seen_tiers and len(alternatives) < 3:
                seen_tiers.add(opt["tier_name"])
                alternatives.append(BoxAlternative(
                    tier_name=opt["tier_name"],
                    carton_id=opt["box_id"],
                    total_cost=opt["t_cost"],
                    utilization_pct=opt["util_pct"],
                    risk_probability=opt["r_prob"],
                    cost_breakdown=opt["cost_breakdown"],
                ))

        total_cart_cost += best_option["t_cost"]

        # -- GenAI / fallback explanation --------------------------------
        explanation, gen_instructions = explainer.generate_summary(
            best_option["box_id"], best_option["r_prob"], best_option["t_cost"],
            best_option["util_pct"], group, request.use_genai, request.model_provider,
        )

        # Contextual label for breakdown
        best_box = best_option["box_id"]
        if any(tag in best_box for tag in ["TRUCK", "MULTI"]):
            detail_label = "Loading & Handling"
        elif any(tag in best_box for tag in ["PALLET", "REEFER"]):
            detail_label = "Pallet & Handling"
        else:
            detail_label = "Packaging"

        carton_reasoning = (
            f"Selected {best_option['box_id']} via {best_option['mode_name']} -- "
            f"most cost-effective valid combination at Rs.{best_option['t_cost']:,.0f}."
        )
        cost_breakdown = (
            f"{detail_label}: Rs.{best_option['m_cost']:,.0f}\n"
            f"Freight: Rs.{best_option['s_cost']:,.0f}\n"
            f"Total: Rs.{best_option['t_cost']:,.0f}"
        )
        risk_reasoning = (
            f"Risk rated at {best_option['r_prob']*100:.1f}% based on material "
            f"density and {100.0 - best_option['util_pct']*100:.1f}% void space."
        )

        # -- Packing instructions (used when GenAI is off) ---------------
        instructions = []
        instructions.append(
            "**Moisture Protection:** Indian transit encounters extreme humidity. "
            "Enclose inner items in a sealed LDPE shrink-wrap layer to prevent "
            "cardboard degradation."
        )
        instructions.append(
            "**Tamper-Evident Sealing:** Seal all outer edges with cross-strapped "
            "BOPP security tape to deter pilferage during multi-node transit."
        )

        if "Fine Art & Antiques" in categories or "Glass & Ceramics" in categories or highest_frag >= 0.85:
            instructions.append(
                "**Shock Absorption:** Use Honeycomb Kraft Paper inserts or "
                "Suspension Packaging instead of standard bubble wrap for "
                "high-impact drop protection."
            )
        if "Precision Electronics" in categories:
            instructions.append(
                "**Heat & Static Defence:** Pack inside ESD bags and include "
                "Silica Gel desiccants to prevent condensation damage in "
                "high-temperature transit (+40 C)."
            )
        if "Perishable Goods" in categories:
            instructions.append(
                "**Thermal Shielding:** Use EPS cooler boxes lined with gel "
                "packs or Phase Change Materials (PCMs) to maintain temperature."
            )

        has_liquid = any(i.is_liquid for i in group)
        has_orientation = any(i.orientation_sensitive for i in group)

        if has_liquid:
            instructions.append(
                "**Leak Prevention:** Double-seal all liquid caps with tape, "
                "place inside a primary ziplock, and surround with absorbent "
                "cellulose wadding."
            )
        if has_orientation:
            instructions.append(
                "**Orientation:** Apply clearly visible 'THIS WAY UP' labels "
                "on all sides of the carton."
            )

        if highest_frag < 0.3:
            instructions.append(
                "**Void Fill:** Pack tight with crushed kraft paper to lock "
                "items in place against vibration."
            )
        else:
            instructions.append(
                "**Immobilisation:** Fragile items must not shift. Fill all "
                "voids with dense, custom-cut EPE foam planks."
            )

        packing_instructions = gen_instructions if gen_instructions else "\n\n".join(instructions)

        shipments.append(ShipmentResponse(
            items=group,
            recommended_carton_id=best_option["box_id"],
            total_estimated_cost=best_option["t_cost"],
            damage_risk_probability=best_option["r_prob"],
            genai_explanation=explanation,
            carton_reasoning=carton_reasoning,
            cost_breakdown=cost_breakdown,
            risk_reasoning=risk_reasoning,
            packing_instructions=packing_instructions,
            transit_mode_advice=best_option["mode_advice"],
            alternatives=alternatives,
        ))

    # ------------------------------------------------------------------
    # 4. Response
    # ------------------------------------------------------------------
    compatibility_warning = ""
    if len(groups) > 1:
        compatibility_warning = (
            "Smart Grouping: Your cart contained incompatible items "
            "(e.g. liquids with electronics, or perishables). "
            "We have automatically split your order into separate safe shipments."
        )

    return RecommendationResponse(
        shipments=shipments,
        total_cart_cost=total_cart_cost,
        compatibility_warning=compatibility_warning,
        source_zone=source_zone,
        destination_zone=dest_zone,
    )
