"""Synthetic cart and SKU generator for Packaging Advisor evaluation.

Generates diverse carts across all 10 canonical product categories,
ground-truth fragility ranges, dimensions, weights, liquid/orientation/
cold-chain flags, and Indian Zone A–E PIN routes.
"""

import os
import random
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.catalog import CATEGORY_NAMES, FRAGILITY_RANGES
from backend.schemas.cart import SKUItem

CATEGORIES = CATEGORY_NAMES

DIM_WEIGHT_RANGES = {
    "General":               (10, 30,  8, 25,  5, 20, 0.5,  4.0),
    "Precision Electronics": (12, 35, 10, 25,  3, 15, 0.3,  3.5),
    "Perishable Goods":      (10, 26, 10, 22,  8, 18, 0.5,  5.0),
    "Fine Art & Antiques":   (15, 42, 12, 32, 10, 28, 0.8,  7.0),
    "Glass & Ceramics":      (10, 25, 10, 22,  8, 22, 0.4,  4.0),
    "Industrial Machinery":  (20, 48, 15, 38, 12, 34, 4.0, 20.0),
    "Textiles & Apparel":    (15, 32, 12, 25,  4, 15, 0.2,  2.0),
    "Pharmaceuticals":       ( 8, 22,  6, 16,  5, 14, 0.1,  2.0),
    "Automotive Parts":      (18, 42, 14, 32, 10, 25, 2.0, 14.0),
    "Books & Media":         (14, 28, 10, 22,  4, 14, 0.3,  3.0),
}

SAMPLE_PINS = [
    "110001",  # Delhi NCR (Zone A/C)
    "201301",  # Noida NCR (Zone A with Delhi)
    "400001",  # Mumbai MMR
    "411001",  # Pune (Zone B with Mumbai)
    "560001",  # Bengaluru
    "600001",  # Chennai
    "700001",  # Kolkata
    "302001",  # Jaipur (Regional / National non-metro)
    "452001",  # Indore
    "781001",  # Guwahati (Zone E)
    "190001",  # Srinagar (Zone E)
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
    fragility = (
        fragility_override
        if fragility_override is not None
        else round(rng.uniform(f_min, f_max), 2)
    )

    l_min, l_max, w_min, w_max, h_min, h_max, wt_min, wt_max = DIM_WEIGHT_RANGES[cat]
    l = round(rng.uniform(l_min, l_max), 1)
    w = round(rng.uniform(w_min, w_max), 1)
    h = round(rng.uniform(h_min, h_max), 1)
    wt = (
        weight_override
        if weight_override is not None
        else round(rng.uniform(wt_min, wt_max), 2)
    )

    is_liquid = False
    if cat == "Perishable Goods" and rng.random() < 0.25:
        is_liquid = True
    elif cat == "Pharmaceuticals" and rng.random() < 0.30:
        is_liquid = True
    elif cat in ("General", "Glass & Ceramics") and rng.random() < 0.10:
        is_liquid = True
    elif cat in ("Industrial Machinery", "Automotive Parts") and rng.random() < 0.10:
        is_liquid = True

    requires_cold_chain = (cat == "Perishable Goods") or (
        cat == "Pharmaceuticals" and rng.random() < 0.25
    )

    orientation_sensitive = False
    if is_liquid:
        orientation_sensitive = True
    elif (
        cat in ("Precision Electronics", "Fine Art & Antiques", "Glass & Ceramics")
        and rng.random() < 0.35
    ):
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
        quantity=1,
        requires_cold_chain=requires_cold_chain,
    )


def generate_cart_scenarios(
    num_carts: int = 200,
    seed: int = 42,
) -> List[dict]:
    """Generate >= 150 synthetic cart scenarios spanning diverse cargo mixes & zones."""
    rng = random.Random(seed)
    carts = []
    item_counter = 1

    for cid in range(num_carts):
        archetype_roll = cid % 6

        if archetype_roll == 0:
            cat = CATEGORIES[cid % len(CATEGORIES)]
            item_count = rng.randint(1, 4)
            cats = [cat] * item_count
        elif archetype_roll == 1:
            item_count = rng.randint(2, 4)
            cats = ["Perishable Goods", "Precision Electronics"]
            if item_count > 2:
                cats.extend(
                    rng.choices(["General", "Pharmaceuticals"], k=item_count - 2)
                )
        elif archetype_roll == 2:
            item_count = rng.randint(2, 4)
            cats = ["Glass & Ceramics", "Industrial Machinery"]
            if item_count > 2:
                cats.extend(
                    rng.choices(
                        ["Fine Art & Antiques", "Precision Electronics"],
                        k=item_count - 2,
                    )
                )
        elif archetype_roll == 3:
            item_count = rng.randint(2, 5)
            cats = rng.choices(
                ["Textiles & Apparel", "Books & Media", "General"], k=item_count
            )
        elif archetype_roll == 4:
            item_count = rng.randint(1, 3)
            cats = ["Industrial Machinery"] * (item_count - 1) + [
                rng.choice(["Automotive Parts", "General"])
            ]
        else:
            item_count = rng.randint(2, 5)
            cats = rng.choices(CATEGORIES, k=item_count)

        items = []
        for cat in cats:
            items.append(generate_sku_item(cat, item_counter, rng=rng))
            item_counter += 1

        src_pin = rng.choice(SAMPLE_PINS)
        if rng.random() < 0.18:
            dst_pin = src_pin[:3] + f"{rng.randint(10, 99):02d}1"
        else:
            dst_pin = rng.choice(SAMPLE_PINS)

        carts.append(
            {
                "cart_id": cid,
                "items": items,
                "source_pin": src_pin,
                "destination_pin": dst_pin,
                "archetype": archetype_roll,
                "category_mix": list(set(cats)),
            }
        )

    return carts


if __name__ == "__main__":
    test_carts = generate_cart_scenarios(150)
    print(f"Generated {len(test_carts)} synthetic cart scenarios successfully.")
