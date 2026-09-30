"""Idempotent startup database seeding for Master SKUs and historical audits."""

from datetime import datetime, timedelta, timezone
import json
from sqlalchemy.orm import Session

from backend.core.catalog import calculate_item_fragility, estimate_item_value_inr
from backend.db.models import MasterSKU, OrderAudit, ShipmentRecord


_SEED_SKUS = [
    ("ELEC-MBP-14", "Enterprise Laptop 14-inch", "Precision Electronics", 35.0, 24.5, 4.5, 1.85, 68000.0, False, False, False),
    ("ELEC-PCB-02", "Server NIC Expansion Module", "Precision Electronics", 22.0, 16.0, 5.0, 0.65, 19500.0, False, False, False),
    ("PHRM-INS-01", "Cold-Chain Biologic Vial Kit", "Pharmaceuticals", 16.0, 12.0, 9.0, 0.85, 12500.0, True, True, True),
    ("PERISH-MNG-05", "Alphonso Export Fresh Box", "Perishable Goods", 24.0, 18.0, 12.0, 2.50, 2400.0, False, True, True),
    ("GLS-LAB-09", "Borosilicate Labware Set", "Glass & Ceramics", 22.0, 18.0, 20.0, 2.10, 4800.0, False, True, False),
    ("ART-BRZ-03", "Chola Bronze Sculpture", "Fine Art & Antiques", 28.0, 22.0, 32.0, 5.40, 42000.0, False, True, False),
    ("MECH-PMP-12", "2HP Centrifugal Industrial Pump", "Industrial Machinery", 38.0, 28.0, 24.0, 16.50, 24000.0, False, True, False),
    ("AUTO-BRK-08", "Vented Disc Brake Rotor Pair", "Automotive Parts", 34.0, 34.0, 12.0, 11.20, 7200.0, False, False, False),
    ("TXT-APP-20", "Corporate Cotton Apparel Bundle", "Textiles & Apparel", 30.0, 24.0, 10.0, 1.40, 2800.0, False, False, False),
    ("BOOK-CAT-15", "Technical Compliance Manual Pack", "Books & Media", 26.0, 19.0, 8.0, 2.20, 1500.0, False, False, False),
    ("LIQ-SOL-07", "Industrial Surface Solvent 1L", "General", 11.0, 11.0, 24.0, 1.35, 750.0, True, True, False),
    ("GEN-KIT-01", "Employee Onboarding Desk Kit", "General", 28.0, 22.0, 10.0, 1.60, 1800.0, False, False, False),
]

_SEED_ORDERS = [
    ("ORD-20261112-1001", "PO-INFOSYS-881", "110001", "560001", "North", "South", "Zone C (Metro-to-Metro Corridor)", "balanced", 2, 3, 4.05, 415.0, 565.0, 295.0, 0.048, 1.85, 0.65, "C-LARGE-99", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Surface", "3-4 days", 38.4, 52.1, 7.0, "delivered_intact", "Arrived intact at Bengaluru hub"),
    ("ORD-20261113-1002", "PO-APOLLO-304", "400001", "411001", "West", "West", "Zone B (Regional / Intra-State)", "lowest_risk", 1, 2, 1.70, 345.0, 480.0, 260.0, 0.036, 2.45, 0.72, "CC-EPS-S", "5-Ply+EPS, BS 12 kgf/cm2, ECT 5.0 kN/m", "Priority Cold Surface", "2-3 days", 31.2, 46.5, 4.0, "delivered_intact", "Cold pack retained 4.5 C"),
    ("ORD-20261114-1003", "PO-NE-MED-119", "600001", "781001", "South", "North-East", "Zone E (Special / North-East & J&K)", "lowest_risk", 1, 1, 0.85, 1020.0, 1460.0, 390.0, 0.064, 3.60, 0.85, "CC-EPS-S", "5-Ply+EPS, BS 12 kgf/cm2, ECT 5.0 kN/m", "Air", "2-3 days", 28.5, 42.0, 4.5, "delivered_intact", "Express Air bypasses Assam surface delay"),
    ("ORD-20261115-1004", "PO-NCR-B2B-42", "110001", "201301", "North", "North", "Zone A (Local / Metro Cluster)", "lowest_cost", 2, 4, 5.20, 142.0, 205.0, 75.0, 0.029, 0.72, 0.24, "C-MED-05", "3-Ply, BS 8 kgf/cm2, ECT 4.0 kN/m", "Last-Mile", "1-2 days", 62.5, 68.0, 6.0, "delivered_intact", "Same-day NCR van delivery"),
    ("ORD-20261116-1005", "PO-TATA-MTR-55", "700001", "500001", "East", "South", "Zone C (Metro-to-Metro Corridor)", "balanced", 1, 1, 16.50, 560.0, 745.0, 250.0, 0.041, 2.35, 0.58, "C-HEAVY-5P", "5-Ply, BS 16 kgf/cm2, ECT 6.8 kN/m", "Surface", "3-4 days", 44.1, 49.0, 19.0, "delivered_intact", "Zero flange deformation"),
    ("ORD-20261117-1006", "PO-JAIPUR-ART-9", "302001", "751001", "West", "East", "Zone D (National Cross-Country)", "lowest_risk", 1, 1, 5.40, 1690.0, 2320.0, 710.0, 0.072, 6.90, 1.45, "C-LARGE-99", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Air", "2 days", 35.8, 59.4, 17.0, "delivered_intact", "ISTA 3cm EPE corner blocks effective"),
    ("ORD-20261118-1007", "PO-MUM-LAB-77", "400001", "560001", "West", "South", "Zone C (Metro-to-Metro Corridor)", "lowest_cost", 2, 2, 3.45, 420.0, 590.0, 180.0, 0.118, 1.62, 0.52, "C-LARGE-99", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Surface", "3-4 days", 33.4, 48.2, 17.0, "damaged_in_transit", "Minor hairline crack on outer beaker"),
    ("ORD-20261119-1008", "PO-ECO-TXT-81", "110001", "400001", "North", "West", "Zone C (Metro-to-Metro Corridor)", "eco_friendly", 2, 6, 8.40, 395.0, 540.0, 115.0, 0.028, 1.28, 0.44, "C-LARGE-99", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Surface", "3-4 days", 64.2, 64.2, 17.0, "delivered_intact", "High density apparel consolidation"),
    ("ORD-20261120-1009", "PO-KASHMIR-19", "110001", "190001", "North", "North", "Zone E (Special / North-East & J&K)", "balanced", 1, 2, 3.70, 1140.0, 1620.0, 620.0, 0.065, 5.20, 1.25, "C-FLAT-04", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Air", "2-3 days", 52.1, 61.0, 7.0, "pending", ""),
    ("ORD-20261121-1010", "PO-AUTO-BLR-64", "560001", "600001", "South", "South", "Zone C (Metro-to-Metro Corridor)", "balanced", 1, 2, 22.40, 740.0, 980.0, 210.0, 0.044, 2.85, 0.78, "C-HEAVY-5P", "5-Ply, BS 16 kgf/cm2, ECT 6.8 kN/m", "Surface", "3-4 days", 42.8, 48.0, 25.0, "pending", ""),
    ("ORD-20261122-1011", "PO-FRESH-GOA-33", "403001", "400001", "West", "West", "Zone D (National Cross-Country)", "balanced", 1, 2, 5.00, 990.0, 1380.0, 210.0, 0.056, 4.80, 1.05, "CC-EPS-M", "5-Ply+EPS, BS 14 kgf/cm2, ECT 5.8 kN/m", "Air", "2 days", 37.5, 49.5, 14.5, "pending", ""),
    ("ORD-20261123-1012", "PO-INDORE-GEN-14", "452001", "700001", "Central", "East", "Zone D (National Cross-Country)", "eco_friendly", 2, 3, 6.10, 485.0, 670.0, 145.0, 0.033, 1.54, 0.51, "C-LARGE-99", "5-Ply, BS 12 kgf/cm2, ECT 5.2 kN/m", "Surface", "4-6 days", 51.0, 56.8, 17.0, "pending", ""),
]


def seed_initial_data(db: Session) -> None:
    """Populate Master SKUs and historical OrderAudit records on fresh database."""
    if db.query(MasterSKU).count() == 0:
        for (
            sku_code,
            name,
            cat,
            l,
            w,
            h,
            wt,
            val,
            is_liq,
            orient,
            cold,
        ) in _SEED_SKUS:
            frag = calculate_item_fragility(cat, l, w, h, wt)
            db.add(
                MasterSKU(
                    sku_code=sku_code,
                    name=name,
                    product_category=cat,
                    length_cm=l,
                    width_cm=w,
                    height_cm=h,
                    weight_kg=wt,
                    fragility_score=frag,
                    declared_value_inr=val if val > 0 else estimate_item_value_inr(cat, wt),
                    is_liquid=is_liq,
                    orientation_sensitive=orient,
                    requires_cold_chain=cold,
                )
            )
        db.commit()

    if db.query(OrderAudit).count() == 0:
        now = datetime.now(timezone.utc)
        for idx, row in enumerate(_SEED_ORDERS):
            (
                order_ref,
                po,
                src_pin,
                dst_pin,
                src_zone,
                dst_zone,
                zone_tier,
                prio,
                n_skus,
                n_units,
                tot_wt,
                direct_cost,
                base_cost,
                exp_dmg,
                risk_prob,
                co2_kg,
                co2_saved,
                carton_id,
                spec_sum,
                mode,
                sla,
                net_util,
                pad_util,
                chg_wt,
                outcome,
                notes,
            ) = row

            created_ts = now - timedelta(days=len(_SEED_ORDERS) - idx)
            savings = round(max(0.0, base_cost - direct_cost), 2)
            landed = round(direct_cost + exp_dmg, 2)

            audit = OrderAudit(
                order_ref=order_ref,
                customer_or_po=po,
                source_pin=src_pin,
                destination_pin=dst_pin,
                source_zone=src_zone,
                destination_zone=dst_zone,
                shipping_zone_tier=zone_tier,
                optimization_priority=prio,
                total_skus=n_skus,
                total_units=n_units,
                total_weight_kg=tot_wt,
                total_direct_cost=direct_cost,
                baseline_unoptimized_cost=base_cost,
                cost_savings_inr=savings,
                total_expected_damage_cost=exp_dmg,
                total_effective_landed_cost=landed,
                total_co2_kg=co2_kg,
                co2_saved_kg=co2_saved,
                created_at=created_ts,
            )
            db.add(audit)
            db.flush()

            badges = ["ISTA 3A CUSHIONED"] if risk_prob > 0.04 else []
            if "CC-EPS" in carton_id:
                badges.append("COLD CHAIN 2-8 C")
            if tot_wt >= 15.0:
                badges.append("HEAVY LIFT (>15 KG)")

            sample_items = [
                {
                    "sku_id": f"{po}-ITEM",
                    "product_category": "General",
                    "length_cm": 24.0,
                    "width_cm": 18.0,
                    "height_cm": 12.0,
                    "weight_kg": round(tot_wt / max(1, n_units), 2),
                    "quantity": n_units,
                    "fragility_score": 0.45,
                }
            ]

            db.add(
                ShipmentRecord(
                    order_id=audit.id,
                    shipment_index=0,
                    group_label="perishables" if "CC-EPS" in carton_id else "general",
                    recommended_carton_id=carton_id,
                    box_count=1,
                    container_spec_summary=spec_sum,
                    transit_mode=mode,
                    estimated_transit_days=sla,
                    net_utilization_pct=net_util,
                    padded_utilization_pct=pad_util,
                    chargeable_weight_kg=chg_wt,
                    direct_cost=direct_cost,
                    expected_damage_cost=exp_dmg,
                    damage_risk_probability=risk_prob,
                    co2_emissions_kg=co2_kg,
                    handling_badges_json=json.dumps(badges),
                    items_json=json.dumps(sample_items),
                    placements_json="[]",
                    packing_instructions="Standard BIS IS 2771 + LDPE monsoon wrap.",
                    actual_outcome=outcome,
                    damage_notes=notes,
                    outcome_recorded_at=(
                        created_ts + timedelta(days=3) if outcome != "pending" else None
                    ),
                )
            )
        db.commit()
