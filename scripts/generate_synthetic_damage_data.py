"""
generate_synthetic_damage_data.py
=================================
Synthetic cart and SKU generator for Packaging Advisor damage evaluation
and model training.

Generates realistic items spanning:
- 8 product categories (General, Electronics, Perishables, Fine Art, Glass, Machinery, Textiles, Pharmaceuticals)
- Ground-truth fragility ranges per category
- Variable item dimensions (length, width, height) and weights
- Liquid flags and orientation sensitivity
- Diverse PIN code routes (source and destination across North, South, East, West zones)

Declared as synthetic, domain-rule grounded data with controlled stochastic noise.
"""

import os
import sys
import random
from typing import List, Tuple, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from backend.schemas.cart import SKUItem

# Product categories used throughout Packaging Advisor
CATEGORIES = [
    "General",
    "Precision Electronics",
    "Perishable Goods",
    "Fine Art & Antiques",
    "Glass & Ceramics",
    "Industrial Machinery",
    "Textiles & Apparel",
    "Pharmaceuticals",
]

# Category -> typical fragility range (min, max)
FRAGILITY_RANGES = {
    "General":                (0.10, 0.40),
    "Precision Electronics":  (0.55, 0.85),
    "Perishable Goods":       (0.40, 0.70),
    "Fine Art & Antiques":    (0.75, 0.98),
    "Glass & Ceramics":       (0.70, 0.95),
    "Industrial Machinery":   (0.15, 0.45),
    "Textiles & Apparel":     (0.05, 0.20),
    "Pharmaceuticals":        (0.35, 0.60),
}

# Category -> typical dimensional bounds (L_min, L_max, W_min, W_max, H_min, H_max, wt_min, wt_max) in cm and kg
DIM_WEIGHT_RANGES = {
    "General":                (10, 30,  8, 25,  5, 20, 0.5,  4.0),
    "Precision Electronics":  (12, 35, 10, 25,  3, 15, 0.3,  3.5),
    "Perishable Goods":       (10, 28, 10, 22,  8, 20, 0.5,  5.0),
    "Fine Art & Antiques":    (15, 45, 12, 35, 10, 30, 0.8,  8.0),
    "Glass & Ceramics":       (10, 25, 10, 22,  8, 22, 0.4,  4.0),
    "Industrial Machinery":   (20, 50, 15, 40, 12, 35, 4.0, 20.0),
    "Textiles & Apparel":     (15, 32, 12, 25,  4, 15, 0.2,  2.0),
    "Pharmaceuticals":        ( 8, 22,  6, 16,  5, 14, 0.1,  2.0),
}

# Representative PIN codes for multi-zone and local shipping routes
SAMPLE_PINS = [
    "110001",  # North (Delhi)
    "110020",  # North (Delhi, same subzone for last-mile testing)
    "560001",  # South (Bangalore)
    "600001",  # South (Chennai)
    "400001",  # West (Mumbai)
    "700001",  # East (Kolkata)
    "900001",  # APS / Remote
]


def generate_sku_item(
    category: str,
    item_id: int,
    rng: Optional[random.Random] = None,
    fragility_override: Optional[float] = None,
    weight_override: Optional[float] = None,
) -> SKUItem:
    """Generate a single realistic SKU item given a category."""
    if rng is None:
        rng = random.Random()

    cat = category if category in FRAGILITY_RANGES else "General"
    f_min, f_max = FRAGILITY_RANGES[cat]
    fragility = fragility_override if fragility_override is not None else round(rng.uniform(f_min, f_max), 2)

    l_min, l_max, w_min, w_max, h_min, h_max, wt_min, wt_max = DIM_WEIGHT_RANGES[cat]
    l = round(rng.uniform(l_min, l_max), 1)
    w = round(rng.uniform(w_min, w_max), 1)
    h = round(rng.uniform(h_min, h_max), 1)
    wt = weight_override if weight_override is not None else round(rng.uniform(wt_min, wt_max), 2)

    # Liquid likelihood by category
    is_liquid = False
    if cat == "Perishable Goods" and rng.random() < 0.25:
        is_liquid = True
    elif cat == "Pharmaceuticals" and rng.random() < 0.35:
        is_liquid = True
    elif cat in ["General", "Glass & Ceramics"] and rng.random() < 0.08:
        is_liquid = True
    elif cat == "Industrial Machinery" and rng.random() < 0.12:
        is_liquid = True

    # Orientation sensitivity: liquids must stay upright; some electronics and art also require it
    orientation_sensitive = False
    if is_liquid:
        orientation_sensitive = True
    elif cat in ["Precision Electronics", "Fine Art & Antiques", "Glass & Ceramics"] and rng.random() < 0.35:
        orientation_sensitive = True

    return SKUItem(
        sku_id=f"SKU-{item_id:04d}",
        product_category=cat,
        length_cm=l,
        width_cm=w,
        height_cm=h,
        weight_kg=wt,
        fragility_score=fragility,
        is_liquid=is_liquid,
        orientation_sensitive=orientation_sensitive,
    )


def generate_cart_scenarios(
    num_carts: int = 200,
    seed: int = 42,
) -> List[dict]:
    """
    Generate >=150 synthetic cart scenarios spanning varied category mixes,
    fragility ranges, weight ranges, item counts, and route zones.
    """
    rng = random.Random(seed)
    carts = []
    item_counter = 1

    for cid in range(num_carts):
        # Determine scenario archetype for broad coverage
        archetype_roll = cid % 6

        if archetype_roll == 0:
            # Homogeneous single-category cart (e.g. all electronics, all textiles, all glass)
            cat = CATEGORIES[cid % len(CATEGORIES)]
            item_count = rng.randint(1, 4)
            cats = [cat] * item_count
        elif archetype_roll == 1:
            # Incompatible mix: perishables + electronics or liquids + electronics
            item_count = rng.randint(2, 4)
            cats = ["Perishable Goods", "Precision Electronics"]
            if item_count > 2:
                cats.extend(rng.choices(["General", "Pharmaceuticals"], k=item_count - 2))
        elif archetype_roll == 2:
            # High-fragility cart (fine art, glass, ceramics)
            item_count = rng.randint(1, 4)
            cats = rng.choices(["Fine Art & Antiques", "Glass & Ceramics", "Precision Electronics"], k=item_count)
        elif archetype_roll == 3:
            # Low-fragility, lightweight or bulk cart (textiles, apparel, light general)
            item_count = rng.randint(2, 5)
            cats = rng.choices(["Textiles & Apparel", "General"], k=item_count)
        elif archetype_roll == 4:
            # Heavy industrial / machinery mix
            item_count = rng.randint(1, 3)
            cats = ["Industrial Machinery"] * (item_count - 1) + [rng.choice(CATEGORIES)]
        else:
            # Completely diverse random multi-item cart
            item_count = rng.randint(2, 5)
            cats = rng.choices(CATEGORIES, k=item_count)

        items = []
        for cat in cats:
            item = generate_sku_item(cat, item_counter, rng=rng)
            items.append(item)
            item_counter += 1

        # Route PIN selection (mix of intra-zone and inter-zone)
        src_pin = rng.choice(SAMPLE_PINS)
        if rng.random() < 0.20:
            # Same subzone for local last-mile testing
            dst_pin = src_pin[:2] + f"{rng.randint(10, 99):02d}1"
        else:
            dst_pin = rng.choice(SAMPLE_PINS)

        carts.append({
            "cart_id": cid,
            "items": items,
            "source_pin": src_pin,
            "destination_pin": dst_pin,
            "archetype": archetype_roll,
            "category_mix": list(set(cats)),
        })

    return carts


if __name__ == "__main__":
    test_carts = generate_cart_scenarios(150)
    print(f"Generated {len(test_carts)} synthetic cart scenarios successfully.")
    print(f"Sample cart #0: {len(test_carts[0]['items'])} items, categories={test_carts[0]['category_mix']}")
