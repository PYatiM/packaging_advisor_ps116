"""Empirical Baseline Evaluation & Statistical Validation for Packaging Advisor.

Evaluates 4 conditions across >= 150 diverse cart scenarios:
1. Proposed Pipeline: Compatibility grouping + ISTA 3D CP-SAT packing +
   Zone A–E Routing + Dynamic Cost Engine + Calibrated ML Damage Predictor.
2. Always-Largest-Box Baseline: Forces C-XL-100 for every shipment group.
3. Simulated Manual Baseline: Bounding-box matching on single largest item
   (ignoring ISTA cushioning, multi-item 3D overlap, and payload caps).
4. No-Grouping Baseline: Packs all cart items into one ambient container.

Outputs:
- results/baseline_comparison.csv
- results/statistical_summary.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from backend.core.catalog import CONTAINER_CATALOG, ORDERED_CONTAINERS
from backend.ml.predictor import DamagePredictor
from backend.optimization.solver import PackingSolver
from backend.schemas.cart import SKUItem
from backend.services.cost_engine import DynamicCostEngine
from backend.services.grouping import group_cart_items
from backend.services.routing_engine import RouteProfile, RoutingEngine
from scripts.generate_synthetic_damage_data import generate_cart_scenarios


def _score_candidate_modes(
    items: List[SKUItem],
    cand: dict,
    route: RouteProfile,
    cost_engine: DynamicCostEngine,
    predictor: DamagePredictor,
    priority: str = "balanced",
) -> dict:
    spec = cand.get("spec")
    modes = cost_engine.get_eligible_modes(cand, route)
    options = []
    for mode_name, base_rate, _advice, _days in modes:
        t_days_min = (
            route.air_days_min if mode_name == "Air" else route.surface_days_min
        )
        r_prob = predictor.predict_damage_risk(
            items,
            cand["utilization_pct"],
            route.distance_multiplier,
            padded_utilization_pct=cand["padded_utilization_pct"],
            cushion_adequacy_ratio=cand["cushion_adequacy_ratio"],
            container_spec=spec,
            transit_mode=mode_name,
            handling_nodes=route.handling_nodes,
            transit_days_min=t_days_min,
            box_count=cand["box_count"],
        )
        pricing = cost_engine.calculate_mode_pricing(
            items, cand, route, mode_name, base_rate
        )
        exp_dmg = round(r_prob * pricing["declared_value_inr"] * 0.30, 2)
        options.append(
            {
                "t_cost": pricing["direct_cost"],
                "r_prob": r_prob,
                "effective_landed_cost": round(pricing["direct_cost"] + exp_dmg, 2),
                "box_vol": cand["box_vol"],
                "util_pct": cand["utilization_pct"],
            }
        )
    return cost_engine.sort_options(options, priority)[0]


def evaluate_condition_proposed(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    solver: PackingSolver,
    cost_engine: DynamicCostEngine,
    cost_matrix: dict,
    predictor: DamagePredictor,
    routing: RoutingEngine,
) -> Dict[str, float]:
    """Condition (a): Proposed Pipeline."""
    groups, _ = group_cart_items(items)
    route = routing.analyze_route(src_pin, dst_pin)
    total_cart_item_vol = sum(
        i.length_cm * i.width_cm * i.height_cm * getattr(i, "quantity", 1)
        for i in items
    )

    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for group_lbl, group_items in groups:
        require_cold = group_lbl == "perishables"
        solver_results = solver.solve_3d_bin_packing(
            group_items, cost_matrix, require_cold_chain=require_cold
        )
        best_opt = None
        ranked = []
        for cand in solver_results:
            ranked.append(
                _score_candidate_modes(
                    group_items, cand, route, cost_engine, predictor, "balanced"
                )
            )
        best_opt = cost_engine.sort_options(ranked, "balanced")[0]

        total_cost += best_opt["t_cost"]
        total_box_vol += best_opt["box_vol"]
        unit_count = sum(getattr(i, "quantity", 1) for i in group_items)
        group_risks.append((unit_count, best_opt["r_prob"]))

    total_units = sum(getattr(i, "quantity", 1) for i in items)
    cart_util = min(0.95, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = (
        sum(cnt * r for cnt, r in group_risks) / total_units if total_units else 0.0
    )
    return {
        "utilization_pct": round(cart_util * 100.0, 2),
        "total_estimated_cost": round(total_cost, 2),
        "damage_risk_probability": round(cart_risk, 4),
    }


def evaluate_condition_always_largest(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    cost_engine: DynamicCostEngine,
    predictor: DamagePredictor,
    routing: RoutingEngine,
) -> Dict[str, float]:
    """Condition (b): Always-Largest-Box Baseline (Force C-XL-100 per group)."""
    groups, _ = group_cart_items(items)
    route = routing.analyze_route(src_pin, dst_pin)
    spec = CONTAINER_CATALOG["C-XL-100"]
    box_vol = spec.inner_volume_cm3
    total_cart_item_vol = sum(
        i.length_cm * i.width_cm * i.height_cm * getattr(i, "quantity", 1)
        for i in items
    )

    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for _, group_items in groups:
        g_vol = sum(
            i.length_cm * i.width_cm * i.height_cm * getattr(i, "quantity", 1)
            for i in group_items
        )
        util = min(0.95, g_vol / box_vol)
        cand = {
            "box_id": spec.container_id,
            "box_count": 1,
            "b_cost": spec.material_cost_inr,
            "box_vol": box_vol,
            "outer_vol": spec.outer_volume_cm3,
            "tare_wt": spec.tare_weight_kg,
            "utilization_pct": util,
            "padded_utilization_pct": min(0.98, util * 1.15),
            "cushion_adequacy_ratio": 1.0,
            "container_dims_cm": list(spec.inner_dims_cm),
            "spec": spec,
            "placements": [],
        }
        best_opt = _score_candidate_modes(
            group_items, cand, route, cost_engine, predictor, "lowest_cost"
        )
        total_cost += best_opt["t_cost"]
        total_box_vol += box_vol
        unit_count = sum(getattr(i, "quantity", 1) for i in group_items)
        group_risks.append((unit_count, best_opt["r_prob"]))

    total_units = sum(getattr(i, "quantity", 1) for i in items)
    cart_util = min(0.95, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = (
        sum(cnt * r for cnt, r in group_risks) / total_units if total_units else 0.0
    )
    return {
        "utilization_pct": round(cart_util * 100.0, 2),
        "total_estimated_cost": round(total_cost, 2),
        "damage_risk_probability": round(cart_risk, 4),
    }


def evaluate_condition_simulated_manual(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    solver: PackingSolver,
    cost_engine: DynamicCostEngine,
    predictor: DamagePredictor,
    routing: RoutingEngine,
) -> Dict[str, float]:
    """Condition (c): Simulated Manual Baseline (single largest-item box, unpadded)."""
    groups, _ = group_cart_items(items)
    route = routing.analyze_route(src_pin, dst_pin)
    total_cart_item_vol = sum(
        i.length_cm * i.width_cm * i.height_cm * getattr(i, "quantity", 1)
        for i in items
    )

    dry_specs = [s for s in ORDERED_CONTAINERS if not s.is_cold_chain]
    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for _, group_items in groups:
        largest_item = max(
            group_items, key=lambda i: i.length_cm * i.width_cm * i.height_cm
        )
        chosen_spec = dry_specs[-1]
        for s in dry_specs:
            bl, bw, bh = s.inner_dims_cm
            if solver._item_fits_box(largest_item, bl, bw, bh):
                chosen_spec = s
                break

        bl, bw, bh = chosen_spec.inner_dims_cm
        b_vol = chosen_spec.inner_volume_cm3
        g_vol = sum(
            i.length_cm * i.width_cm * i.height_cm * getattr(i, "quantity", 1)
            for i in group_items
        )
        g_wt = sum(i.weight_kg * getattr(i, "quantity", 1) for i in group_items)

        units_raw = solver._expand_units(group_items, use_padding=False)
        units_padded = solver._expand_units(group_items, use_padding=True)
        fits_physically = (
            g_vol <= b_vol
            and g_wt <= chosen_spec.max_payload_kg
            and solver._cpsat_pack(units_raw, bl, bw, bh) is not None
        )

        g_util = min(0.95, g_vol / b_vol)
        cushion_ratio = (
            solver._compute_unpadded_cushion_ratio(units_padded, chosen_spec)
            if fits_physically
            else 0.20
        )

        cand = {
            "box_id": chosen_spec.container_id,
            "box_count": 1,
            "b_cost": chosen_spec.material_cost_inr,
            "box_vol": b_vol,
            "outer_vol": chosen_spec.outer_volume_cm3,
            "tare_wt": chosen_spec.tare_weight_kg,
            "utilization_pct": g_util,
            "padded_utilization_pct": g_util,
            "cushion_adequacy_ratio": cushion_ratio,
            "container_dims_cm": list(chosen_spec.inner_dims_cm),
            "spec": chosen_spec,
            "placements": [],
        }
        best_opt = _score_candidate_modes(
            group_items, cand, route, cost_engine, predictor, "lowest_cost"
        )
        r_prob = best_opt["r_prob"]
        if not fits_physically:
            overfill = max(g_vol / b_vol, g_wt / max(1.0, chosen_spec.max_payload_kg))
            r_prob = min(0.99, max(0.75, r_prob + 0.30 * max(0.0, overfill - 1.0)))

        total_cost += best_opt["t_cost"]
        total_box_vol += b_vol
        unit_count = sum(getattr(i, "quantity", 1) for i in group_items)
        group_risks.append((unit_count, r_prob))

    total_units = sum(getattr(i, "quantity", 1) for i in items)
    cart_util = min(1.0, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = (
        sum(cnt * r for cnt, r in group_risks) / total_units if total_units else 0.0
    )
    return {
        "utilization_pct": round(cart_util * 100.0, 2),
        "total_estimated_cost": round(total_cost, 2),
        "damage_risk_probability": round(cart_risk, 4),
    }


def evaluate_condition_no_grouping(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    solver: PackingSolver,
    cost_engine: DynamicCostEngine,
    cost_matrix: dict,
    predictor: DamagePredictor,
    routing: RoutingEngine,
) -> Dict[str, float]:
    """Condition (d): No-Grouping Baseline (all items packed together without segregation)."""
    route = routing.analyze_route(src_pin, dst_pin)
    solver_results = solver.solve_3d_bin_packing(
        items, cost_matrix, require_cold_chain=False
    )
    ranked = [
        _score_candidate_modes(items, cand, route, cost_engine, predictor, "lowest_cost")
        for cand in solver_results
    ]
    best_opt = cost_engine.sort_options(ranked, "lowest_cost")[0]

    return {
        "utilization_pct": round(best_opt["util_pct"] * 100.0, 2),
        "total_estimated_cost": round(best_opt["t_cost"], 2),
        "damage_risk_probability": round(best_opt["r_prob"], 4),
    }


def run_evaluation(num_carts: int = 200, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    print(f"Generating {num_carts} diverse synthetic cart scenarios (seed={seed})...")
    carts = generate_cart_scenarios(num_carts=num_carts, seed=seed)

    solver = PackingSolver()
    cost_engine = DynamicCostEngine()
    cost_matrix = cost_engine.get_current_rates()
    predictor = DamagePredictor()
    routing = RoutingEngine()

    records = []
    print("Evaluating 4 packaging conditions across all carts...")

    for idx, cart in enumerate(carts):
        cid = cart["cart_id"]
        items = cart["items"]
        src = cart["source_pin"]
        dst = cart["destination_pin"]

        evaluators = [
            (
                "proposed",
                evaluate_condition_proposed(
                    items, src, dst, solver, cost_engine, cost_matrix, predictor, routing
                ),
            ),
            (
                "always_largest",
                evaluate_condition_always_largest(
                    items, src, dst, cost_engine, predictor, routing
                ),
            ),
            (
                "simulated_manual",
                evaluate_condition_simulated_manual(
                    items, src, dst, solver, cost_engine, predictor, routing
                ),
            ),
            (
                "no_grouping",
                evaluate_condition_no_grouping(
                    items, src, dst, solver, cost_engine, cost_matrix, predictor, routing
                ),
            ),
        ]
        for cond, res in evaluators:
            records.append(
                {
                    "cart_id": cid,
                    "condition": cond,
                    "utilization_pct": res["utilization_pct"],
                    "total_estimated_cost": res["total_estimated_cost"],
                    "damage_risk_probability": res["damage_risk_probability"],
                }
            )

        if (idx + 1) % 50 == 0 or (idx + 1) == num_carts:
            print(f"  Processed {idx + 1}/{num_carts} carts...")

    df_records = pd.DataFrame(records)
    metrics = ["utilization_pct", "total_estimated_cost", "damage_risk_probability"]
    conditions = ["proposed", "always_largest", "simulated_manual", "no_grouping"]

    summary_stats = {}
    for cond in conditions:
        sub = df_records[df_records["condition"] == cond]
        summary_stats[cond] = {
            m: {"mean": float(np.mean(sub[m])), "std": float(np.std(sub[m]))}
            for m in metrics
        }

    wilcoxon_results = {}
    ours_df = df_records[df_records["condition"] == "proposed"].sort_values("cart_id")

    for baseline in ["always_largest", "simulated_manual", "no_grouping"]:
        base_df = df_records[df_records["condition"] == baseline].sort_values("cart_id")
        wilcoxon_results[baseline] = {}
        for m in metrics:
            diff = ours_df[m].values - base_df[m].values
            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                try:
                    w_res = wilcoxon(ours_df[m].values, base_df[m].values)
                    stat, p_val = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    stat, p_val = 0.0, 1.0
            wilcoxon_results[baseline][m] = {"statistic": stat, "p_value": p_val}

    full_summary = {
        "conditions": summary_stats,
        "wilcoxon_tests_vs_proposed": wilcoxon_results,
        "scenario_count": num_carts,
        "seed": seed,
    }
    return df_records, full_summary


def print_markdown_table(summary: dict):
    cond_map = [
        ("Proposed Pipeline (Ours)", "proposed", "-"),
        ("Always-Largest-Box Baseline", "always_largest", None),
        ("Simulated Manual Baseline", "simulated_manual", None),
        ("No-Grouping Baseline", "no_grouping", None),
    ]

    print("\n### Baseline Comparison Evaluation\n")
    headers = [
        "Method",
        "Avg Utilization %",
        "Avg Cost",
        "Avg Damage Risk",
        "p-value vs Ours",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for display_name, key, default_p in cond_map:
        u_mean = summary["conditions"][key]["utilization_pct"]["mean"]
        c_mean = summary["conditions"][key]["total_estimated_cost"]["mean"]
        r_mean = summary["conditions"][key]["damage_risk_probability"]["mean"]

        if default_p is not None:
            p_str = default_p
        else:
            p_cost = summary["wilcoxon_tests_vs_proposed"][key]["total_estimated_cost"]["p_value"]
            p_risk = summary["wilcoxon_tests_vs_proposed"][key]["damage_risk_probability"]["p_value"]
            max_p = max(p_cost, p_risk)
            if max_p < 0.001:
                p_str = "p < 0.001"
            elif max_p < 0.01:
                p_str = "p < 0.01"
            elif max_p < 0.05:
                p_str = "p < 0.05"
            else:
                p_str = f"p = {max_p:.3f}"

        print(
            f"| {display_name} | {u_mean:.1f}% | Rs. {c_mean:,.2f} | {r_mean:.3f} | {p_str} |"
        )
    print(
        "\n*Note: p-values derived from paired two-sided Wilcoxon signed-rank tests across identical cart scenarios.*\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Packaging Advisor Baseline Evaluation")
    parser.add_argument("--num-carts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "baseline_comparison.csv")
    json_path = os.path.join(results_dir, "statistical_summary.json")

    df_results, stat_summary = run_evaluation(num_carts=args.num_carts, seed=args.seed)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Artifact Saved] Recorded {len(df_results)} rows to {csv_path}")

    with open(json_path, "w") as f:
        json.dump(stat_summary, f, indent=2)
    print(f"[Artifact Saved] Statistical summary written to {json_path}")

    print_markdown_table(stat_summary)

    if args.ablations:
        from scripts.run_ablations import main as run_ablations_main

        print("\n" + "=" * 70)
        print("CHAINING ABLATION EXPERIMENTS")
        print("=" * 70)
        run_ablations_main()


if __name__ == "__main__":
    main()
