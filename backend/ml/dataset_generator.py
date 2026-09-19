"""
Synthetic Dataset Generator for Packaging Damage-Risk Classification
=====================================================================

Generates labeled training data using domain-grounded rules with
controlled Gaussian noise.  Each row represents a single shipment
scenario with features derived from the SKU / box / route properties
already present in the Packaging Advisor.

**Label definition**
    damaged = 1  if the shipment suffered transit damage
    damaged = 0  otherwise

The ground-truth probability is computed from an interpretable risk
model (fragility, void-space, weight density, liquid flag, orientation
sensitivity, distance multiplier, item count), then thresholded with
stochastic noise to produce a binary label.

>>> THIS IS A SYNTHETIC, RULE-PERTURBED DATASET WITH NOISE. <<<
"""

import numpy as np
import pandas as pd
from typing import Optional


# Reproducibility
_RNG_SEED = 42

# Product categories used in the advisor
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

# Category → typical fragility range (low, high)
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


def generate_dataset(
    n_samples: int = 10_000,
    seed: Optional[int] = _RNG_SEED,
) -> pd.DataFrame:
    """Return a DataFrame with *n_samples* rows of synthetic shipment data.

    Features
    --------
    max_fragility        : float [0, 1]   – highest fragility among items
    void_space_pct       : float [0, 1]   – fraction of box volume unused
    total_weight_kg      : float > 0      – aggregate item weight
    item_count           : int   >= 1     – number of SKUs in the shipment
    has_liquid           : int   {0, 1}   – any liquid items?
    orientation_sensitive: int   {0, 1}   – any orientation-locked items?
    distance_multiplier  : float [1, 3]   – zone-to-zone shipping distance
    weight_density       : float          – total_weight / box_volume_litres
    category_encoded     : int            – ordinal-encoded product category

    Label
    -----
    damaged : int {0, 1}
    """
    rng = np.random.default_rng(seed)

    # --- Sample raw features ------------------------------------------------
    categories = rng.choice(CATEGORIES, size=n_samples)

    fragility = np.array([
        rng.uniform(*FRAGILITY_RANGES[c]) for c in categories
    ])

    void_space = rng.beta(a=2.0, b=5.0, size=n_samples)       # skewed low
    total_weight = rng.lognormal(mean=1.5, sigma=1.2, size=n_samples)
    total_weight = np.clip(total_weight, 0.1, 28_000.0)

    item_count = rng.poisson(lam=3, size=n_samples) + 1        # >= 1

    has_liquid = rng.binomial(1, 0.12, size=n_samples)
    orientation_sens = rng.binomial(1, 0.18, size=n_samples)

    # Distance multiplier mirrors the zone matrix in router.py
    distance_mult = rng.choice(
        [1.0, 1.2, 1.8, 2.0, 2.5, 3.0],
        p=[0.10, 0.25, 0.20, 0.20, 0.15, 0.10],
        size=n_samples,
    )

    # Approximate box volume (litres) — derived so void_space makes sense
    item_vol = rng.lognormal(mean=2.0, sigma=1.0, size=n_samples)
    item_vol = np.clip(item_vol, 0.5, 100_000.0)
    box_vol = item_vol / (1.0 - void_space + 1e-9)
    weight_density = total_weight / (box_vol + 1e-9)

    # Ordinal encoding for category
    cat_map = {c: i for i, c in enumerate(CATEGORIES)}
    cat_encoded = np.array([cat_map[c] for c in categories])

    # --- Ground-truth risk model ---------------------------------------------
    # Calibrated so ~25-30 % of shipments are "damaged" (realistic baseline
    # for a logistics scenario without protective measures).
    risk = (
        0.30 * fragility
        + 0.25 * void_space
        + 0.10 * np.tanh(total_weight / 50.0)
        + 0.08 * has_liquid
        + 0.05 * orientation_sens
        + 0.07 * (distance_mult / 3.0)
        + 0.05 * np.log1p(item_count) / np.log1p(20)
        + 0.10 * np.clip(weight_density / 5.0, 0, 1)
    )

    # Add heteroscedastic noise (more noise at mid-range risk)
    noise_scale = 0.12 * np.sqrt(risk * (1 - risk) + 0.01)
    risk_noisy = risk + rng.normal(0, noise_scale, size=n_samples)
    risk_noisy = np.clip(risk_noisy, 0.0, 1.0)

    # Threshold to binary label
    damaged = (risk_noisy > 0.42).astype(int)

    # --- Assemble DataFrame --------------------------------------------------
    df = pd.DataFrame({
        "max_fragility":         np.round(fragility, 4),
        "void_space_pct":        np.round(void_space, 4),
        "total_weight_kg":       np.round(total_weight, 2),
        "item_count":            item_count,
        "has_liquid":            has_liquid,
        "orientation_sensitive": orientation_sens,
        "distance_multiplier":   distance_mult,
        "weight_density":        np.round(weight_density, 4),
        "category_encoded":      cat_encoded,
        "product_category":      categories,   # kept for interpretability
        "damaged":               damaged,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=10_000)
    out = "backend/ml/data/damage_dataset.csv"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} samples → {out}")
    print(f"Class distribution:\n{df['damaged'].value_counts(normalize=True)}")
