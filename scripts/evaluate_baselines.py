"""
evaluate_baselines.py
=====================
Empirical Baseline Evaluation & Statistical Validation for Packaging Advisor.

Evaluates 4 conditions across >=150 diverse cart scenarios:
1. Proposed Pipeline: Smart compatibility grouping + OR-Tools CP-SAT 3D bin packing +
   Dynamic Cost Engine + ML Damage Predictor.
2. Always-Largest-Box Baseline: Forces C-XL-100 container for every shipment group.
3. Simulated Manual Baseline: Heuristic bounding-box matching (picks the smallest box
   fitting the single largest item, ignoring multi-item volumetric packing constraints).
4. No-Grouping Baseline: Skips compatibility grouping, packing all cart items as a single shipment.

Outputs:
- results/baseline_comparison.csv: granular per-cart per-condition metric recordings.
- results/statistical_summary.json: mean +/- std and paired Wilcoxon signed-rank tests.
- stdout: Markdown summary table matching specified column order.
"""

import os
import sys
import json
import random
from typing import List, Tuple, Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from backend.core.config import settings
from backend.schemas.cart import SKUItem
from backend.optimization.solver import PackingSolver
from backend.services.cost_engine import DynamicCostEngine
from backend.ml.predictor import DamagePredictor
from scripts.generate_synthetic_damage_data import generate_cart_scenarios, CATEGORIES, FRAGILITY_RANGES

# Illustrative prototype assumptions as of 2025-01-15, calibrated broadly to
# private-courier per-kg pricing for small parcels in India; not taken from
# any single published rate card.
AIR_BASE_RATE_PER_KG = settings.AIR_BASE_RATE_PER_KG
SURFACE_BASE_RATE_PER_KG = settings.SURFACE_BASE_RATE_PER_KG

# -----------------------------------------------------------------------------
# Core Distance & Pricing Matrix (synchronized with backend/api/router.py)
# -----------------------------------------------------------------------------
DIST_MATRIX = {
    "North":   {"North": 1.2, "South": 2.5, "East": 2.0, "West": 1.8, "APS": 3.0, "Unknown": 2.0},
    "South":   {"North": 2.5, "South": 1.2, "East": 1.8, "West": 1.8, "APS": 3.0, "Unknown": 2.0},
    "East":    {"North": 2.0, "South": 1.8, "East": 1.2, "West": 2.5, "APS": 3.0, "Unknown": 2.0},
    "West":    {"North": 1.8, "South": 1.8, "East": 2.5, "West": 1.2, "APS": 3.0, "Unknown": 2.0},
    "APS":     {"North": 3.0, "South": 3.0, "East": 3.0, "West": 3.0, "APS": 1.2, "Unknown": 3.0},
    "Unknown": {"North": 2.0, "South": 2.0, "East": 2.0, "West": 2.0, "APS": 3.0, "Unknown": 2.0},
}


def get_zone(pin: str) -> str:
    """Determine Indian postal zone from the leading digit of the PIN code."""
    first_digit = str(pin).strip()[0] if str(pin).strip() else "1"
    if first_digit in ["1", "2"]:
        return "North"
    elif first_digit in ["3", "4"]:
        return "West"
    elif first_digit in ["5", "6"]:
        return "South"
    elif first_digit in ["7", "8"]:
        return "East"
    elif first_digit == "9":
        return "APS"
    return "Unknown"


def compute_distance_multiplier(src_pin: str, dst_pin: str) -> float:
    """Compute lane distance multiplier according to zone routing rules."""
    src = str(src_pin).strip() if src_pin else "110001"
    dst = str(dst_pin).strip() if dst_pin else "560001"
    if len(src) >= 2 and len(dst) >= 2 and src[:2] == dst[:2]:
        return 1.0
    s_zone = get_zone(src)
    d_zone = get_zone(dst)
    return DIST_MATRIX.get(s_zone, {}).get(d_zone, 2.0)


def group_cart_items(items: List[SKUItem]) -> List[Tuple[str, List[SKUItem]]]:
    """
    Apply compatibility grouping rules matching backend/api/router.py:
    1. Perishable Goods (requires cold-chain / temperature control)
    2. Liquids (non-perishable liquids requiring containment)
    3. Precision Electronics (fragile, static/shock sensitive)
    4. General (ambient dry cargo, apparel, machinery, ceramics)
    """
    perishables, liquids, electronics, general = [], [], [], []
    for item in items:
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
    return groups


def get_valid_transit_modes(items: List[SKUItem], is_same_subzone: bool) -> List[Tuple[str, float]]:
    """Determine available transit modes and base rates (Rs/kg).

    These illustrative prototype assumptions are as of 2025-01-15 and can be
    overridden through the shared settings module.
    """
    if is_same_subzone:
        return [("Last-Mile", 25.0)]
    cats = set(i.product_category for i in items)
    if "Perishable Goods" in cats:
        return [("Refrigerated Road", 20.0), ("Air", AIR_BASE_RATE_PER_KG)]
    return [("Surface", SURFACE_BASE_RATE_PER_KG), ("Air", AIR_BASE_RATE_PER_KG)]


def calculate_container_cost(
    box_id: str,
    box_vol: float,
    group_wt: float,
    cost_matrix: dict,
    dist_mult: float,
    valid_modes: List[Tuple[str, float]],
) -> float:
    """Calculate the lowest total shipping + material cost across valid modes."""
    vol_wt = box_vol / 5000.0
    charge_wt = max(group_wt, vol_wt)

    b_costs = cost_matrix.get(box_id)
    if not b_costs:
        if "MULTI-TRUCK" in box_id:
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
    best_t_cost = float("inf")

    for mode_name, base_rate in valid_modes:
        if b_costs.get("shipping_flat"):
            s_cost = b_costs.get("flat_rate", 50000.0) * dist_mult
        else:
            s_cost = charge_wt * base_rate * dist_mult
        t_cost = m_cost + s_cost
        if t_cost < best_t_cost:
            best_t_cost = t_cost

    return best_t_cost


# -----------------------------------------------------------------------------
# Evaluation Pipeline for the 4 Conditions
# -----------------------------------------------------------------------------
def evaluate_condition_proposed(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    solver: PackingSolver,
    cost_matrix: dict,
    predictor: DamagePredictor,
) -> Dict[str, float]:
    """Condition (a): Proposed Pipeline (Grouping + CP-SAT Solver + ML Predictor + Dynamic Cost Engine)."""
    groups = group_cart_items(items)
    is_same_subzone = (len(src_pin) >= 2 and len(dst_pin) >= 2 and src_pin[:2] == dst_pin[:2])
    dist_mult = compute_distance_multiplier(src_pin, dst_pin)
    total_cart_item_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in items)

    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for group_lbl, group_items in groups:
        solver_results = solver.solve_3d_bin_packing(group_items, cost_matrix)
        if group_lbl == "perishables":
            reefer = [r for r in solver_results if "REEFER" in r["box_id"]]
            if reefer:
                solver_results = reefer

        modes = get_valid_transit_modes(group_items, is_same_subzone)
        group_wt = sum(i.weight_kg for i in group_items)

        best_option = None
        best_cost = float("inf")
        for res in solver_results:
            c = calculate_container_cost(res["box_id"], res["box_vol"], group_wt, cost_matrix, dist_mult, modes)
            if c < best_cost:
                best_cost = c
                best_option = res

        total_cost += best_cost
        total_box_vol += best_option["box_vol"]
        util_pct = best_option["utilization_pct"]
        r_prob = predictor.predict_damage_risk(group_items, util_pct)
        group_risks.append((len(group_items), r_prob))

    cart_util = min(0.95, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = sum(count * r for count, r in group_risks) / len(items) if items else 0.0

    return {
        "utilization_pct": round(cart_util * 100.0, 2),
        "total_estimated_cost": round(total_cost, 2),
        "damage_risk_probability": round(cart_risk, 4),
    }


def evaluate_condition_always_largest(
    items: List[SKUItem],
    src_pin: str,
    dst_pin: str,
    cost_matrix: dict,
    predictor: DamagePredictor,
) -> Dict[str, float]:
    """Condition (b): Always-Largest-Box Baseline (Force C-XL-100 for every shipment group)."""
    groups = group_cart_items(items)
    is_same_subzone = (len(src_pin) >= 2 and len(dst_pin) >= 2 and src_pin[:2] == dst_pin[:2])
    dist_mult = compute_distance_multiplier(src_pin, dst_pin)
    total_cart_item_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in items)

    box_id = "C-XL-100"
    box_vol = 100.0 * 100.0 * 100.0  # 1,000,000 cm3
    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for _, group_items in groups:
        modes = get_valid_transit_modes(group_items, is_same_subzone)
        group_wt = sum(i.weight_kg for i in group_items)
        group_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in group_items)

        cost = calculate_container_cost(box_id, box_vol, group_wt, cost_matrix, dist_mult, modes)
        total_cost += cost
        total_box_vol += box_vol

        g_util = min(0.95, group_vol / box_vol)
        r_prob = predictor.predict_damage_risk(group_items, g_util)
        group_risks.append((len(group_items), r_prob))

    cart_util = min(0.95, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = sum(count * r for count, r in group_risks) / len(items) if items else 0.0

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
    cost_matrix: dict,
    predictor: DamagePredictor,
) -> Dict[str, float]:
    """
    Condition (c): Simulated Manual Baseline.
    Picks the smallest container fitting the single largest item's bounding box,
    ignoring multi-item volumetric packing constraints. Overpacked/overweight
    packages incur realistic physical structural crush and damage penalties.
    """
    groups = group_cart_items(items)
    is_same_subzone = (len(src_pin) >= 2 and len(dst_pin) >= 2 and src_pin[:2] == dst_pin[:2])
    dist_mult = compute_distance_multiplier(src_pin, dst_pin)
    total_cart_item_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in items)

    total_cost = 0.0
    total_box_vol = 0.0
    group_risks = []

    for _, group_items in groups:
        modes = get_valid_transit_modes(group_items, is_same_subzone)
        group_wt = sum(i.weight_kg for i in group_items)
        group_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in group_items)

        # Pick smallest box fitting only the single largest item's bounding box
        largest_item = max(group_items, key=lambda i: i.length_cm * i.width_cm * i.height_cm)
        chosen_box = None
        for b in solver.boxes:
            box_id, bl, bw, bh, b_wt, b_cost = b
            if solver._item_fits_box(largest_item, bl, bw, bh):
                chosen_box = b
                break

        if not chosen_box:
            # Fallback to custom crate for oversized items
            chosen_box = ("CUSTOM-CRATE", 100, 100, 100, 50.0, 500.0)

        b_id, bl, bw, bh, b_wt, _ = chosen_box
        b_vol = bl * bw * bh

        cost = calculate_container_cost(b_id, b_vol, group_wt, cost_matrix, dist_mult, modes)
        total_cost += cost
        total_box_vol += b_vol

        # Check physical feasibility (ignored by manual packer)
        fits_physically = (
            group_vol <= b_vol and
            group_wt <= b_wt and
            (len(group_items) == 1 or solver._cpsat_fits(group_items, bl, bw, bh))
        )

        g_util = min(1.0, group_vol / b_vol)
        if fits_physically:
            r_prob = predictor.predict_damage_risk(group_items, min(0.95, g_util))
        else:
            # Box cannot physically accommodate the items without bursting/crushing
            base_risk = predictor.predict_damage_risk(group_items, 0.95)
            overfill_ratio = max(group_vol / b_vol, group_wt / max(1.0, b_wt))
            r_prob = min(0.99, max(0.80, base_risk + 0.35 * (overfill_ratio - 1.0)))

        group_risks.append((len(group_items), r_prob))

    cart_util = min(1.0, total_cart_item_vol / total_box_vol) if total_box_vol > 0 else 0.0
    cart_risk = sum(count * r for count, r in group_risks) / len(items) if items else 0.0

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
    cost_matrix: dict,
    predictor: DamagePredictor,
) -> Dict[str, float]:
    """Condition (d): No-Grouping Baseline (Skip compatibility grouping, pack everything as one shipment)."""
    is_same_subzone = (len(src_pin) >= 2 and len(dst_pin) >= 2 and src_pin[:2] == dst_pin[:2])
    dist_mult = compute_distance_multiplier(src_pin, dst_pin)
    total_cart_item_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in items)
    total_wt = sum(i.weight_kg for i in items)

    modes = get_valid_transit_modes(items, is_same_subzone)
    solver_results = solver.solve_3d_bin_packing(items, cost_matrix)

    best_option = None
    best_cost = float("inf")
    for res in solver_results:
        c = calculate_container_cost(res["box_id"], res["box_vol"], total_wt, cost_matrix, dist_mult, modes)
        if c < best_cost:
            best_cost = c
            best_option = res

    box_vol = best_option["box_vol"]
    cart_util = best_option["utilization_pct"]
    cart_risk = predictor.predict_damage_risk(items, cart_util)

    return {
        "utilization_pct": round(cart_util * 100.0, 2),
        "total_estimated_cost": round(best_cost, 2),
        "damage_risk_probability": round(cart_risk, 4),
    }


# -----------------------------------------------------------------------------
# Main Evaluation Runner
# -----------------------------------------------------------------------------
def run_evaluation(num_carts: int = 200, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    """Run full comparative benchmark across all carts and conditions."""
    print(f"Generating {num_carts} diverse synthetic cart scenarios (seed={seed})...")
    carts = generate_cart_scenarios(num_carts=num_carts, seed=seed)

    solver = PackingSolver()
    cost_matrix = DynamicCostEngine().get_current_rates()
    predictor = DamagePredictor()

    records = []
    print("Evaluating 4 packaging conditions across all carts directly via backend logic...")

    for idx, cart in enumerate(carts):
        cid = cart["cart_id"]
        items = cart["items"]
        src = cart["source_pin"]
        dst = cart["destination_pin"]

        # 1. Proposed pipeline
        res_a = evaluate_condition_proposed(items, src, dst, solver, cost_matrix, predictor)
        records.append({
            "cart_id": cid,
            "condition": "proposed",
            "utilization_pct": res_a["utilization_pct"],
            "total_estimated_cost": res_a["total_estimated_cost"],
            "damage_risk_probability": res_a["damage_risk_probability"],
        })

        # 2. Always-largest box (C-XL-100)
        res_b = evaluate_condition_always_largest(items, src, dst, cost_matrix, predictor)
        records.append({
            "cart_id": cid,
            "condition": "always_largest",
            "utilization_pct": res_b["utilization_pct"],
            "total_estimated_cost": res_b["total_estimated_cost"],
            "damage_risk_probability": res_b["damage_risk_probability"],
        })

        # 3. Simulated manual baseline
        res_c = evaluate_condition_simulated_manual(items, src, dst, solver, cost_matrix, predictor)
        records.append({
            "cart_id": cid,
            "condition": "simulated_manual",
            "utilization_pct": res_c["utilization_pct"],
            "total_estimated_cost": res_c["total_estimated_cost"],
            "damage_risk_probability": res_c["damage_risk_probability"],
        })

        # 4. No-grouping baseline
        res_d = evaluate_condition_no_grouping(items, src, dst, solver, cost_matrix, predictor)
        records.append({
            "cart_id": cid,
            "condition": "no_grouping",
            "utilization_pct": res_d["utilization_pct"],
            "total_estimated_cost": res_d["total_estimated_cost"],
            "damage_risk_probability": res_d["damage_risk_probability"],
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == num_carts:
            print(f"  Processed {idx + 1}/{num_carts} carts...")

    df_records = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Statistical Analysis: Mean, Std, and Paired Wilcoxon Signed-Rank Tests
    # -------------------------------------------------------------------------
    metrics = ["utilization_pct", "total_estimated_cost", "damage_risk_probability"]
    conditions = ["proposed", "always_largest", "simulated_manual", "no_grouping"]

    summary_stats = {}
    for cond in conditions:
        sub = df_records[df_records["condition"] == cond]
        summary_stats[cond] = {}
        for m in metrics:
            summary_stats[cond][m] = {
                "mean": float(np.mean(sub[m])),
                "std": float(np.std(sub[m])),
            }

    # Paired Wilcoxon tests vs Proposed
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
                except Exception as exc:
                    stat, p_val = 0.0, 1.0
            wilcoxon_results[baseline][m] = {
                "statistic": stat,
                "p_value": p_val,
            }

    full_statistical_summary = {
        "conditions": summary_stats,
        "wilcoxon_tests_vs_proposed": wilcoxon_results,
        "scenario_count": num_carts,
        "seed": seed,
    }

    return df_records, full_statistical_summary


def print_markdown_table(summary: dict):
    """
    Print markdown-formatted summary table matching required column order:
    Method | Avg Utilization % | Avg Cost | Avg Damage Risk | p-value vs Ours
    """
    cond_map = [
        ("Proposed Pipeline (Ours)", "proposed", "-"),
        ("Always-Largest-Box Baseline", "always_largest", None),
        ("Simulated Manual Baseline", "simulated_manual", None),
        ("No-Grouping Baseline", "no_grouping", None),
    ]

    print("\n### Baseline Comparison Evaluation\n")
    headers = ["Method", "Avg Utilization %", "Avg Cost", "Avg Damage Risk", "p-value vs Ours"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for display_name, key, default_p in cond_map:
        u_mean = summary["conditions"][key]["utilization_pct"]["mean"]
        c_mean = summary["conditions"][key]["total_estimated_cost"]["mean"]
        r_mean = summary["conditions"][key]["damage_risk_probability"]["mean"]

        if default_p is not None:
            p_str = default_p
        else:
            # Report maximum / representative p-value vs Ours across primary comparison metrics
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

        print(f"| {display_name} | {u_mean:.1f}% | Rs. {c_mean:,.2f} | {r_mean:.3f} | {p_str} |")

    print("\n*Note: p-values derived from paired two-sided Wilcoxon signed-rank tests (scipy.stats.wilcoxon) across identical cart scenarios.*\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Packaging Advisor Baseline Evaluation")
    parser.add_argument("--num-carts", type=int, default=200, help="Number of cart scenarios (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--ablations", action="store_true", help="Also execute full ablation studies suite")
    args = parser.parse_args()

    # Save directory setup
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "baseline_comparison.csv")
    json_path = os.path.join(results_dir, "statistical_summary.json")

    # Run evaluation with >= 150 scenarios (default 200 carts)
    df_results, stat_summary = run_evaluation(num_carts=args.num_carts, seed=args.seed)

    # Save outputs
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Artifact Saved] Recorded {len(df_results)} rows to {csv_path}")

    with open(json_path, "w") as f:
        json.dump(stat_summary, f, indent=2)
    print(f"[Artifact Saved] Statistical summary written to {json_path}")

    # Print markdown summary table to stdout matching requested column order
    print_markdown_table(stat_summary)

    # Optional chained ablation run
    if args.ablations:
        from scripts.run_ablations import main as run_ablations_main
        print("\n" + "=" * 70)
        print("CHAINING ABLATION EXPERIMENTS")
        print("=" * 70)
        run_ablations_main()


if __name__ == "__main__":
    main()
