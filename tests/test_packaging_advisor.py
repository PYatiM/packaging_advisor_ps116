"""Comprehensive unit & integration tests for Packaging Advisor PS116."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from backend.core.catalog import (
    CATEGORY_NAMES,
    CONTAINER_CATALOG,
    ORDERED_CONTAINERS,
    compute_cushion_pad_cm,
)
from backend.main import app
from backend.optimization.solver import PackingSolver
from backend.schemas.cart import SKUItem
from backend.services.cost_engine import DynamicCostEngine
from backend.services.grouping import group_cart_items
from backend.services.routing_engine import RoutingEngine

client = TestClient(app)


def _assert_valid_3d_packing(placements: list, dims_cm: list) -> None:
    """Assert containment and zero pairwise 3D overlap per bin."""
    L, W, H = dims_cm
    eps = 1e-4
    by_bin = {}
    for p in placements:
        assert p["x_cm"] >= -eps
        assert p["y_cm"] >= -eps
        assert p["z_cm"] >= -eps
        assert p["x_cm"] + p["dx_cm"] <= L + eps
        assert p["y_cm"] + p["dy_cm"] <= W + eps
        assert p["z_cm"] + p["dz_cm"] <= H + eps
        by_bin.setdefault(p["bin_index"], []).append(p)

    for bin_items in by_bin.values():
        n = len(bin_items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = bin_items[i], bin_items[j]
                sep_x = (a["x_cm"] + a["dx_cm"] <= b["x_cm"] + eps) or (
                    b["x_cm"] + b["dx_cm"] <= a["x_cm"] + eps
                )
                sep_y = (a["y_cm"] + a["dy_cm"] <= b["y_cm"] + eps) or (
                    b["y_cm"] + b["dy_cm"] <= a["y_cm"] + eps
                )
                sep_z = (a["z_cm"] + a["dz_cm"] <= b["z_cm"] + eps) or (
                    b["z_cm"] + b["dz_cm"] <= a["z_cm"] + eps
                )
                assert sep_x or sep_y or sep_z, f"3D overlap detected between {a} and {b}"


# ------------------------------------------------------------------
# 1. PIN-Code & Zone A-E Routing Tests
# ------------------------------------------------------------------
def test_routing_zone_a_delhi_noida():
    r = RoutingEngine().analyze_route("110001", "201301")
    assert r.zone_code == "A"
    assert r.distance_multiplier == 1.0
    assert r.is_same_subzone is True
    assert r.handling_nodes == 2


def test_routing_zone_b_regional():
    r_intra = RoutingEngine().analyze_route("400001", "411001")
    assert r_intra.zone_code == "B"
    assert r_intra.distance_multiplier == 1.30
    assert r_intra.handling_nodes == 3

    r_state = RoutingEngine().analyze_route("560001", "570001")
    assert r_state.zone_code == "B"
    assert r_state.distance_multiplier == 1.30


def test_routing_zone_c_metro_corridor():
    r = RoutingEngine().analyze_route("110001", "560001")
    assert r.zone_code == "C"
    assert r.distance_multiplier == 1.65
    assert r.handling_nodes == 4


def test_routing_zone_d_national():
    r_adj = RoutingEngine().analyze_route("124001", "492001")  # Haryana -> Chhattisgarh
    assert r_adj.zone_code == "D"
    assert r_adj.distance_multiplier == 2.05
    assert r_adj.handling_nodes == 5

    r_opp = RoutingEngine().analyze_route("302001", "751001")  # Rajasthan -> Odisha
    assert r_opp.zone_code == "D"
    assert r_opp.distance_multiplier == 2.35


def test_routing_zone_e_special_regions():
    for special_pin in ("781001", "190001", "900001"):
        r = RoutingEngine().analyze_route("110001", special_pin)
        assert r.zone_code == "E"
        assert r.distance_multiplier == 2.85
        assert r.is_oda_special is True
        assert r.handling_nodes == 7


def test_routing_invalid_pin_fallback():
    r = RoutingEngine().analyze_route("bad", "")
    assert r.source_pin == "110001"
    assert r.destination_pin == "560001"
    assert r.zone_code == "C"


# ------------------------------------------------------------------
# 2. Smart Cargo Compatibility Grouping Tests
# ------------------------------------------------------------------
def test_grouping_isolates_incompatible_cargo():
    items = [
        SKUItem(
            sku_id="PER-1", product_category="Perishable Goods",
            length_cm=15, width_cm=12, height_cm=10, weight_kg=1.0, fragility_score=0.55,
        ),
        SKUItem(
            sku_id="LIQ-1", product_category="General",
            length_cm=10, width_cm=10, height_cm=18, weight_kg=1.2, fragility_score=0.25,
            is_liquid=True,
        ),
        SKUItem(
            sku_id="ELC-1", product_category="Precision Electronics",
            length_cm=25, width_cm=18, height_cm=5, weight_kg=1.5, fragility_score=0.75,
        ),
        SKUItem(
            sku_id="GLS-1", product_category="Glass & Ceramics",
            length_cm=15, width_cm=15, height_cm=20, weight_kg=1.8, fragility_score=0.88,
        ),
        SKUItem(
            sku_id="MCH-1", product_category="Industrial Machinery",
            length_cm=30, width_cm=25, height_cm=20, weight_kg=14.0, fragility_score=0.25,
        ),
    ]
    groups, reasons = group_cart_items(items)
    labels = [lbl for lbl, _ in groups]
    assert "perishables" in labels
    assert "liquids" in labels
    assert "electronics" in labels
    assert "fragile_delicate" in labels
    assert "general" in labels
    assert len(reasons) >= 3


def test_grouping_consolidates_compatible_dry_items():
    items = [
        SKUItem(
            sku_id="TXT-1", product_category="Textiles & Apparel",
            length_cm=25, width_cm=20, height_cm=6, weight_kg=0.7, fragility_score=0.10,
        ),
        SKUItem(
            sku_id="BK-1", product_category="Books & Media",
            length_cm=22, width_cm=15, height_cm=5, weight_kg=0.9, fragility_score=0.18,
        ),
        SKUItem(
            sku_id="GEN-1", product_category="General",
            length_cm=12, width_cm=10, height_cm=8, weight_kg=0.5, fragility_score=0.22,
        ),
    ]
    groups, reasons = group_cart_items(items)
    assert len(groups) == 1
    assert groups[0][0] == "general"
    assert reasons == []


def test_grouping_cold_chain_flag_on_pharma():
    pharma = SKUItem(
        sku_id="PH-COLD", product_category="Pharmaceuticals",
        length_cm=14, width_cm=10, height_cm=8, weight_kg=0.6,
        fragility_score=0.50, requires_cold_chain=True,
    )
    groups, _ = group_cart_items([pharma])
    assert len(groups) == 1
    assert groups[0][0] == "perishables"


# ------------------------------------------------------------------
# 3. 3D CP-SAT Packing, ISTA Cushioning, Orientation & Multi-Box Tests
# ------------------------------------------------------------------
def test_ista_cushion_expands_required_box_size():
    solver = PackingSolver()
    low_frag = SKUItem(
        sku_id="SOFT", product_category="Textiles & Apparel",
        length_cm=18, width_cm=13, height_cm=10, weight_kg=0.8, fragility_score=0.12,
    )
    high_frag = SKUItem(
        sku_id="GLASS", product_category="Glass & Ceramics",
        length_cm=18, width_cm=13, height_cm=10, weight_kg=0.8, fragility_score=0.85,
    )
    res_low = solver.solve_3d_bin_packing([low_frag])
    res_high = solver.solve_3d_bin_packing([high_frag])
    assert res_low[0]["box_id"] == "C-SMALL-01"
    assert res_high[0]["box_id"] == "C-MED-05"
    assert compute_cushion_pad_cm(0.85, 0.8) == 2.5


def test_orientation_sensitive_preserves_height_on_z():
    solver = PackingSolver()
    # Height is 26 cm; length & width are 12x12 cm.
    # C-MED-05 is 32x25x22 -> cannot hold 26 cm upright! Must upgrade to C-LARGE-99 (50x40x38).
    tall_upright = SKUItem(
        sku_id="BOTTLE", product_category="General",
        length_cm=12, width_cm=12, height_cm=26, weight_kg=1.5,
        fragility_score=0.15, orientation_sensitive=True,
    )
    res = solver.solve_3d_bin_packing([tall_upright])
    assert res[0]["box_id"] == "C-LARGE-99"
    assert res[0]["placements"][0]["dz_cm"] == 26.0


def test_quantity_expansion_and_3d_non_overlap():
    solver = PackingSolver()
    multi = SKUItem(
        sku_id="WIDGET", product_category="Precision Electronics",
        length_cm=14, width_cm=12, height_cm=8, weight_kg=1.1,
        fragility_score=0.75, quantity=3,
    )
    extra = SKUItem(
        sku_id="BOOK", product_category="Books & Media",
        length_cm=20, width_cm=15, height_cm=5, weight_kg=0.9,
        fragility_score=0.15, quantity=1,
    )
    res = solver.solve_3d_bin_packing([multi, extra])
    chosen = res[0]
    assert len(chosen["placements"]) == 4
    _assert_valid_3d_packing(chosen["placements"], chosen["container_dims_cm"])


def test_multi_carton_splitting_for_large_parcel_batch():
    solver = PackingSolver()
    bulky = SKUItem(
        sku_id="GEAR", product_category="Automotive Parts",
        length_cm=38, width_cm=30, height_cm=24, weight_kg=14.0,
        fragility_score=0.30, quantity=4,
    )
    res = solver.solve_3d_bin_packing([bulky])
    multi_splits = [c for c in res if c["box_count"] > 1]
    assert len(multi_splits) >= 1
    for split_cand in multi_splits:
        assert 2 <= split_cand["box_count"] <= 5
        assert len(split_cand["placements"]) == 4
        _assert_valid_3d_packing(split_cand["placements"], split_cand["container_dims_cm"])


# ------------------------------------------------------------------
# 4. Cold-Chain Pricing & PTL / Mode Logic Tests
# ------------------------------------------------------------------
def test_small_perishable_uses_passive_cold_parcel_not_reefer():
    payload = {
        "items": [
            {
                "sku_id": "CAKE-1",
                "product_category": "Perishable Goods",
                "length_cm": 16.0,
                "width_cm": 12.0,
                "height_cm": 9.0,
                "weight_kg": 1.0,
                "fragility_score": 0.55,
            }
        ],
        "source_pin": "110001",
        "destination_pin": "560001",
        "use_genai": False,
        "optimization_priority": "balanced",
    }
    resp = client.post("/api/v1/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    shipment = data["shipments"][0]
    assert shipment["recommended_carton_id"] == "CC-EPS-S"
    assert shipment["total_estimated_cost"] < 1500.0


def test_heavy_perishable_uses_ptl_reefer():
    payload = {
        "items": [
            {
                "sku_id": "DAIRY-BULK",
                "product_category": "Perishable Goods",
                "length_cm": 100.0,
                "width_cm": 80.0,
                "height_cm": 60.0,
                "weight_kg": 200.0,
                "fragility_score": 0.45,
            }
        ],
        "source_pin": "110001",
        "destination_pin": "560001",
        "use_genai": False,
    }
    resp = client.post("/api/v1/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["shipments"][0]["recommended_carton_id"] == "PTL-REEFER"


def test_passive_cold_parcel_no_slow_surface_on_zone_d():
    route_d = RoutingEngine().analyze_route("302001", "751001")
    engine = DynamicCostEngine()
    spec = CONTAINER_CATALOG["CC-EPS-S"]
    cand = {"box_id": spec.container_id, "spec": spec}
    modes = [m[0] for m in engine.get_eligible_modes(cand, route_d)]
    assert "Surface" not in modes
    assert "Priority Cold Surface" not in modes
    assert "Air" in modes


def test_ptl_pallet_cft_minimum_billable_weight():
    route_c = RoutingEngine().analyze_route("110001", "560001")
    engine = DynamicCostEngine()
    spec = CONTAINER_CATALOG["PALLET-STD"]
    item = SKUItem(
        sku_id="LIGHT-PALLET", product_category="General",
        length_cm=80, width_cm=80, height_cm=80, weight_kg=10.0, fragility_score=0.2,
    )
    cand = {
        "box_id": spec.container_id,
        "box_count": 1,
        "b_cost": spec.material_cost_inr,
        "box_vol": spec.inner_volume_cm3,
        "outer_vol": spec.outer_volume_cm3,
        "tare_wt": spec.tare_weight_kg,
        "utilization_pct": 0.30,
        "padded_utilization_pct": 0.30,
        "spec": spec,
    }
    pricing = engine.calculate_mode_pricing([item], cand, route_c, "PTL Surface", 11.0)
    assert pricing["chargeable_wt_kg"] >= 50.0


# ------------------------------------------------------------------
# 5. Optimization Priority & FastAPI Endpoint Integration Tests
# ------------------------------------------------------------------
def test_api_health_and_catalog():
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    cat = client.get("/api/v1/catalog")
    assert cat.status_code == 200
    body = cat.json()
    assert len(body["categories"]) == len(CATEGORY_NAMES) == 10
    assert len(body["containers"]) == len(ORDERED_CONTAINERS) == 14


def test_optimization_priority_lowest_cost_vs_lowest_risk():
    base_req = {
        "items": [
            {
                "sku_id": "ART-VALUABLE",
                "product_category": "Fine Art & Antiques",
                "length_cm": 30.0,
                "width_cm": 25.0,
                "height_cm": 12.0,
                "weight_kg": 3.0,
                "fragility_score": 0.90,
                "declared_value_inr": 65000.0,
            }
        ],
        "source_pin": "302001",
        "destination_pin": "751001",
        "use_genai": False,
    }

    r_cost = client.post("/api/v1/recommend", json={**base_req, "optimization_priority": "lowest_cost"})
    r_risk = client.post("/api/v1/recommend", json={**base_req, "optimization_priority": "lowest_risk"})
    assert r_cost.status_code == 200
    assert r_risk.status_code == 200

    s_cost = r_cost.json()["shipments"][0]
    s_risk = r_risk.json()["shipments"][0]
    assert s_cost["total_estimated_cost"] <= s_risk["total_estimated_cost"]
    assert s_risk["damage_risk_probability"] <= s_cost["damage_risk_probability"]


def test_api_recommend_returns_complete_upgraded_schema():
    payload = {
        "items": [
            {
                "sku_id": "LAPTOP-99",
                "product_category": "Precision Electronics",
                "length_cm": 35.0,
                "width_cm": 24.0,
                "height_cm": 4.5,
                "weight_kg": 1.8,
                "fragility_score": 0.78,
                "quantity": 2,
                "declared_value_inr": 50000.0,
            }
        ],
        "source_pin": "110001",
        "destination_pin": "560001",
        "use_genai": False,
        "optimization_priority": "balanced",
    }
    res = client.post("/api/v1/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["shipping_zone_tier"].startswith("Zone C")
    assert data["total_expected_damage_cost"] > 0
    assert data["total_effective_landed_cost"] > data["total_cart_cost"]

    sh = data["shipments"][0]
    assert len(sh["placements"]) == 2
    assert len(sh["container_inner_dims_cm"]) == 3
    assert sh["chargeable_weight_kg"] >= sh["actual_gross_weight_kg"]
    assert sh["estimated_transit_days"] != ""
    assert "Ply" in sh["container_spec_summary"] or "Custom" in sh["container_spec_summary"]
    assert len(sh["alternatives"]) >= 1
    _assert_valid_3d_packing(sh["placements"], sh["container_inner_dims_cm"])
