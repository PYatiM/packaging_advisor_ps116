"""End-to-end integration tests for Database CRUD, Custom Containers, Manifests, QA & Analytics."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_seeded_skus_populated_on_startup():
    res = client.get("/api/v1/skus")
    assert res.status_code == 200
    skus = res.json()
    assert len(skus) >= 12
    assert any(s["sku_code"] == "ELEC-MBP-14" for s in skus)


def test_seeded_orders_and_analytics_populated():
    orders = client.get("/api/v1/orders").json()
    assert len(orders) >= 12

    analytics = client.get("/api/v1/analytics/summary").json()
    assert analytics["kpis"]["total_orders"] >= 12
    assert analytics["kpis"]["total_shipments"] >= 12
    assert analytics["kpis"]["master_sku_count"] >= 12
    assert analytics["kpis"]["total_savings_inr"] > 0
    assert analytics["kpis"]["total_co2_saved_kg"] > 0
    assert len(analytics["zone_breakdown"]) >= 1
    assert len(analytics["container_usage"]) >= 1
    assert len(analytics["recent_trend"]) >= 12
    assert analytics["qa_calibration"]["evaluated_shipments"] >= 6


def test_master_sku_create_and_search():
    payload = {
        "sku_code": "TEST-SKU-900",
        "name": "Precision Laser Interferometer",
        "product_category": "Precision Electronics",
        "length_cm": 22.0,
        "width_cm": 16.0,
        "height_cm": 10.0,
        "weight_kg": 1.4,
        "declared_value_inr": 45000.0,
        "is_liquid": False,
        "orientation_sensitive": True,
        "requires_cold_chain": False,
    }
    res = client.post("/api/v1/skus", json=payload)
    assert res.status_code == 200
    created = res.json()
    assert created["sku_code"] == "TEST-SKU-900"
    assert created["fragility_score"] > 0.5

    res_search = client.get("/api/v1/skus", params={"search": "Interferometer"})
    assert res_search.status_code == 200
    assert any(s["sku_code"] == "TEST-SKU-900" for s in res_search.json())


def test_master_sku_category_filter_and_update():
    res_cat = client.get("/api/v1/skus", params={"category": "Precision Electronics"})
    assert res_cat.status_code == 200
    assert all(s["product_category"] == "Precision Electronics" for s in res_cat.json())

    res_put = client.put(
        "/api/v1/skus/TEST-SKU-900",
        json={"weight_kg": 2.1, "name": "Precision Laser v2"},
    )
    assert res_put.status_code == 200
    assert res_put.json()["name"] == "Precision Laser v2"
    assert res_put.json()["weight_kg"] == 2.1


def test_master_sku_delete_and_not_found():
    res_del = client.delete("/api/v1/skus/TEST-SKU-900")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

    assert client.delete("/api/v1/skus/TEST-SKU-900").status_code == 404
    assert client.put("/api/v1/skus/NO-SUCH-SKU", json={"weight_kg": 1.0}).status_code == 404


def test_custom_container_builtin_id_conflict():
    conflict = client.post(
        "/api/v1/containers/custom",
        json={
            "container_id": "C-SMALL-01",
            "name": "Duplicate Built-in",
            "inner_l_cm": 20,
            "inner_w_cm": 15,
            "inner_h_cm": 12,
            "max_payload_kg": 3,
            "material_cost_inr": 25,
        },
    )
    assert conflict.status_code == 409


def test_custom_container_participates_in_catalog_and_recommend():
    res_reg = client.post(
        "/api/v1/containers/custom",
        json={
            "container_id": "WH-SLIM-44",
            "name": "Warehouse Slim Laptop Mailer",
            "tier_class": "parcel",
            "inner_l_cm": 44.0,
            "inner_w_cm": 30.0,
            "inner_h_cm": 12.0,
            "tare_weight_kg": 0.45,
            "max_payload_kg": 8.0,
            "ply_rating": "5-Ply",
            "burst_strength_kgf_cm2": 12.0,
            "ect_kn_m": 5.5,
            "material_cost_inr": 65.0,
        },
    )
    assert res_reg.status_code == 200

    custom_list = client.get("/api/v1/containers/custom").json()
    assert any(c["container_id"] == "WH-SLIM-44" for c in custom_list)
    catalog = client.get("/api/v1/catalog").json()
    assert any(
        c["container_id"] == "WH-SLIM-44" and c["is_custom"] is True
        for c in catalog["containers"]
    )

    rec = client.post(
        "/api/v1/recommend",
        json={
            "items": [
                {
                    "sku_id": "SLIM-PANEL",
                    "product_category": "Books & Media",
                    "length_cm": 41.0,
                    "width_cm": 27.0,
                    "height_cm": 8.0,
                    "weight_kg": 1.8,
                    "fragility_score": 0.18,
                }
            ],
            "source_pin": "110001",
            "destination_pin": "560001",
            "optimization_priority": "lowest_cost",
            "persist_history": False,
        },
    )
    assert rec.status_code == 200
    assert rec.json()["shipments"][0]["recommended_carton_id"] == "WH-SLIM-44"

    assert client.delete("/api/v1/containers/custom/WH-SLIM-44").status_code == 200
    assert client.delete("/api/v1/containers/custom/WH-SLIM-44").status_code == 404


def test_recommend_eco_friendly_prefers_low_carbon_surface():
    payload = {
        "items": [
            {
                "sku_id": "ECO-APPAREL-1",
                "product_category": "Textiles & Apparel",
                "length_cm": 28.0,
                "width_cm": 22.0,
                "height_cm": 10.0,
                "weight_kg": 1.5,
                "fragility_score": 0.12,
                "quantity": 2,
            }
        ],
        "source_pin": "110001",
        "destination_pin": "400001",
        "optimization_priority": "eco_friendly",
        "persist_history": False,
    }
    res = client.post("/api/v1/recommend", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["shipments"][0]["transit_mode"] == "Surface"
    assert body["total_co2_kg"] > 0.0
    assert body["co2_saved_kg"] >= 0.0


def test_recommend_persists_order_and_returns_handling_badges():
    res = client.post(
        "/api/v1/recommend",
        json={
            "items": [
                {
                    "sku_id": "ECO-APPAREL-1",
                    "product_category": "Textiles & Apparel",
                    "length_cm": 28.0,
                    "width_cm": 22.0,
                    "height_cm": 10.0,
                    "weight_kg": 1.5,
                    "fragility_score": 0.12,
                    "quantity": 2,
                },
                {
                    "sku_id": "LIQ-BOTTLE-2",
                    "product_category": "General",
                    "length_cm": 10.0,
                    "width_cm": 10.0,
                    "height_cm": 20.0,
                    "weight_kg": 1.1,
                    "fragility_score": 0.25,
                    "is_liquid": True,
                    "orientation_sensitive": True,
                },
            ],
            "source_pin": "110001",
            "destination_pin": "400001",
            "customer_or_po": "PO-CORP-9901",
            "optimization_priority": "eco_friendly",
            "persist_history": True,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["order_ref"].startswith("ORD-")
    assert body["baseline_unoptimized_cost"] >= body["total_cart_cost"]
    assert len(body["shipments"]) == 2
    liq_ship = [s for s in body["shipments"] if s["group_label"] == "liquids"][0]
    assert "LIQUID CONTAINMENT" in liq_ship["handling_badges"]
    assert "THIS WAY UP" in liq_ship["handling_badges"]


def test_order_detail_endpoint_returns_persisted_shipments():
    orders = client.get("/api/v1/orders").json()
    po_orders = [o for o in orders if o["customer_or_po"] == "PO-CORP-9901"]
    assert len(po_orders) >= 1
    order_ref = po_orders[0]["order_ref"]

    detail = client.get(f"/api/v1/orders/{order_ref}")
    assert detail.status_code == 200
    od = detail.json()
    assert od["customer_or_po"] == "PO-CORP-9901"
    assert len(od["shipments"]) == 2


def test_order_detail_not_found():
    assert client.get("/api/v1/orders/ORD-DOES-NOT-EXIST").status_code == 404


def test_order_manifest_generates_html_and_markdown():
    orders = client.get("/api/v1/orders").json()
    po_orders = [o for o in orders if o["customer_or_po"] == "PO-CORP-9901"]
    order_ref = po_orders[0]["order_ref"]

    man_res = client.get(f"/api/v1/orders/{order_ref}/manifest")
    assert man_res.status_code == 200
    man = man_res.json()
    assert "PO-CORP-9901" in man["html_manifest"]
    assert "ECO-APPAREL-1" in man["markdown_manifest"]
    assert "LIQUID CONTAINMENT" in man["markdown_manifest"]


def test_shipment_qa_outcome_updates_analytics_calibration():
    orders = client.get("/api/v1/orders").json()
    po_orders = [o for o in orders if o["customer_or_po"] == "PO-CORP-9901"]
    order_ref = po_orders[0]["order_ref"]
    detail = client.get(f"/api/v1/orders/{order_ref}").json()
    ship_id = detail["shipments"][0]["id"]

    before_intact = client.get("/api/v1/analytics/summary").json()["qa_calibration"]["intact_count"]
    qa_res = client.patch(
        f"/api/v1/shipments/{ship_id}/outcome",
        json={"actual_outcome": "delivered_intact", "damage_notes": "Clean delivery"},
    )
    assert qa_res.status_code == 200
    assert qa_res.json()["actual_outcome"] == "delivered_intact"
    assert qa_res.json()["outcome_recorded_at"] is not None

    after_intact = client.get("/api/v1/analytics/summary").json()["qa_calibration"]["intact_count"]
    assert after_intact == before_intact + 1


def test_shipment_qa_outcome_not_found():
    res = client.patch(
        "/api/v1/shipments/9999999/outcome",
        json={"actual_outcome": "delivered_intact", "damage_notes": ""},
    )
    assert res.status_code == 404


def test_huggingface_qwen_small_model_switching_and_fallback(monkeypatch):
    from types import SimpleNamespace
    from backend.genai.service import ExplainerService
    from backend.schemas.cart import SKUItem

    captured_calls = []

    class _DummyClient:
        def __init__(self, token=None):
            self.token = token

        def chat_completion(self, model, messages, max_tokens=900, temperature=0.3):
            captured_calls.append({"model": model, "messages": messages})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="[ANALYSIS]\n### Why This Container\nCompact.\n[INSTRUCTIONS]\n1. Seal seams."
                        )
                    )
                ]
            )

    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_token_123")
    monkeypatch.setattr("backend.genai.service.InferenceClient", _DummyClient)

    svc = ExplainerService()
    item = SKUItem(
        sku_id="QWN-TEST",
        product_category="Precision Electronics",
        length_cm=30.0,
        width_cm=22.0,
        height_cm=5.0,
        weight_kg=1.4,
        fragility_score=0.75,
    )

    # Default Qwen 2.5 3B Instruct
    analysis, instr = svc.generate_summary(
        "C-FLAT-04",
        0.04,
        250.0,
        0.55,
        [item],
        use_genai=True,
        model_provider="Hugging Face",
    )
    assert captured_calls[0]["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert "**AI Insights (Hugging Face • Qwen/Qwen2.5-3B-Instruct):**" in analysis
    assert "Seal seams" in instr

    # Explicit Qwen 2.5 1.5B Instruct override
    analysis_small, _ = svc.generate_summary(
        "C-FLAT-04",
        0.04,
        250.0,
        0.55,
        [item],
        use_genai=True,
        model_provider="Hugging Face",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert captured_calls[1]["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "**AI Insights (Hugging Face • Qwen/Qwen2.5-1.5B-Instruct):**" in analysis_small

    # Clean deterministic fallback when key is absent
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    svc_no_key = ExplainerService()
    fallback_md, fallback_instr = svc_no_key.generate_summary(
        "C-FLAT-04",
        0.04,
        250.0,
        0.55,
        [item],
        use_genai=True,
        model_provider="Hugging Face",
        model_name="Qwen/Qwen2.5-3B-Instruct",
    )
    assert "### Why This Container" in fallback_md
    assert fallback_instr == ""


def test_pricing_monotonic_across_zones_and_realistic_parcel_costs():
    tshirt = {
        "sku_id": "TSHIRT-05",
        "product_category": "Textiles & Apparel",
        "length_cm": 22.0,
        "width_cm": 16.0,
        "height_cm": 4.0,
        "weight_kg": 0.5,
        "fragility_score": 0.12,
    }
    r_local = client.post(
        "/api/v1/recommend",
        json={"items": [tshirt], "source_pin": "110001", "destination_pin": "201301", "persist_history": False},
    ).json()
    r_metro = client.post(
        "/api/v1/recommend",
        json={"items": [tshirt], "source_pin": "110001", "destination_pin": "560001", "persist_history": False},
    ).json()
    r_special = client.post(
        "/api/v1/recommend",
        json={"items": [tshirt], "source_pin": "110001", "destination_pin": "781001", "persist_history": False},
    ).json()

    c_local = r_local["total_cart_cost"]
    c_metro = r_metro["total_cart_cost"]
    c_special = r_special["total_cart_cost"]
    assert c_local < c_metro < c_special
    assert 60.0 <= c_metro <= 160.0
    assert 5.0 <= r_metro["cost_savings_pct"] <= 65.0

    # Cushioned 1.8 kg laptop in C-FLAT-04 Surface should have realistic cost & low calibrated risk
    laptop = {
        "sku_id": "LAPTOP-REAL",
        "product_category": "Precision Electronics",
        "length_cm": 35.0,
        "width_cm": 24.0,
        "height_cm": 4.5,
        "weight_kg": 1.8,
        "fragility_score": 0.75,
        "declared_value_inr": 60000.0,
    }
    r_lap = client.post(
        "/api/v1/recommend",
        json={
            "items": [laptop],
            "source_pin": "110001",
            "destination_pin": "560001",
            "optimization_priority": "lowest_cost",
            "persist_history": False,
        },
    ).json()
    s_lap = r_lap["shipments"][0]
    assert s_lap["recommended_carton_id"] == "C-FLAT-04"
    assert 150.0 <= s_lap["total_estimated_cost"] <= 350.0
    assert s_lap["damage_risk_probability"] < 0.15
