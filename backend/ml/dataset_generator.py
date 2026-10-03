"""Physics-grounded synthetic dataset generator for damage-risk modeling.

Generates labeled shipment scenarios spanning 10 product categories,
IS 2771 board ECT grades, ISTA cushion adequacy ratios, Indian Zone A–E
handling-node counts, transit vibration factors, and cargo incompatibility
hazards.
"""

from typing import Optional
import numpy as np
import pandas as pd

from backend.core.catalog import (
    CATEGORY_MAP,
    CATEGORY_NAMES,
    FRAGILITY_RANGES,
)

_RNG_SEED = 42

FEATURE_COLS = [
    "max_fragility",
    "mean_fragility",
    "unprotected_void_pct",
    "cushion_adequacy_ratio",
    "total_weight_kg",
    "item_count",
    "has_liquid",
    "orientation_sensitive",
    "distance_multiplier",
    "handling_nodes",
    "weight_to_limit_ratio",
    "board_ect_kn_m",
    "transit_vibration_factor",
    "incompatibility_hazard",
    "category_encoded",
]

_ZONE_ROUTE_PAIRS = [
    (1.00, 2),  # Zone A
    (1.30, 3),  # Zone B
    (1.65, 4),  # Zone C
    (2.05, 5),  # Zone D adjacent
    (2.35, 5),  # Zone D opposite
    (2.85, 7),  # Zone E special / ODA
]
_ZONE_PROBS = [0.15, 0.22, 0.23, 0.16, 0.14, 0.10]

_ECT_VALUES = [3.2, 4.0, 5.0, 5.2, 5.8, 6.8, 7.2, 9.6, 12.4, 15.0]
_VIBRATION_VALUES = [0.45, 0.55, 0.65, 0.75, 1.00]
_VIBRATION_PROBS = [0.22, 0.15, 0.10, 0.08, 0.45]


def generate_dataset(
    n_samples: int = 12_000,
    seed: Optional[int] = _RNG_SEED,
) -> pd.DataFrame:
    """Generate *n_samples* physics-grounded synthetic shipment rows."""
    rng = np.random.default_rng(seed)

    categories = rng.choice(CATEGORY_NAMES, size=n_samples)
    max_fragility = np.array(
        [rng.uniform(*FRAGILITY_RANGES[c]) for c in categories],
        dtype=float,
    )

    item_count = np.clip(rng.poisson(lam=2.6, size=n_samples) + 1, 1, 25)
    mean_fragility = np.where(
        item_count == 1,
        max_fragility,
        max_fragility * rng.uniform(0.65, 1.0, size=n_samples),
    )

    # ~64% properly packed shipments vs ~36% packaging failure modes
    abused = rng.random(n_samples) < 0.36

    cushion_adequacy_ratio = np.where(
        max_fragility < 0.25,
        1.0,
        np.where(
            abused & (rng.random(n_samples) < 0.55),
            rng.uniform(0.12, 0.48, size=n_samples),
            rng.uniform(0.88, 1.25, size=n_samples),
        ),
    )

    unprotected_void_pct = np.where(
        abused & (rng.random(n_samples) < 0.45),
        rng.uniform(0.55, 0.90, size=n_samples),
        rng.beta(a=1.8, b=5.5, size=n_samples),
    )
    total_weight_kg = np.clip(
        rng.lognormal(mean=1.4, sigma=1.25, size=n_samples),
        0.2,
        2500.0,
    )

    has_liquid = rng.binomial(1, 0.14, size=n_samples)
    orientation_sensitive = np.where(
        has_liquid == 1,
        1,
        rng.binomial(1, 0.20, size=n_samples),
    )

    route_indices = rng.choice(len(_ZONE_ROUTE_PAIRS), p=_ZONE_PROBS, size=n_samples)
    distance_multiplier = np.array([_ZONE_ROUTE_PAIRS[i][0] for i in route_indices])
    handling_nodes = np.array([_ZONE_ROUTE_PAIRS[i][1] for i in route_indices])

    board_ect_kn_m = rng.choice(_ECT_VALUES, size=n_samples)

    weight_to_limit_ratio = np.where(
        abused & (rng.random(n_samples) < 0.35),
        rng.uniform(0.95, 1.40, size=n_samples),
        rng.uniform(0.05, 0.85, size=n_samples),
    )

    transit_vibration_factor = rng.choice(
        _VIBRATION_VALUES, p=_VIBRATION_PROBS, size=n_samples
    )

    incompatibility_hazard = np.where(
        abused & (rng.random(n_samples) < 0.55),
        rng.choice([0.35, 0.50, 0.70, 0.85], size=n_samples),
        0.0,
    )

    category_encoded = np.array([CATEGORY_MAP[c] for c in categories], dtype=int)

    # Calibrated logistic damage physics with true Bernoulli draws
    overload = np.maximum(0.0, weight_to_limit_ratio - 0.85)
    low_ect_factor = np.maximum(0.0, (6.0 - board_ect_kn_m) / 3.0)

    logit = (
        -4.45
        + 1.35 * max_fragility
        + 0.40 * mean_fragility
        + 3.60 * max_fragility * np.maximum(0.0, 0.85 - cushion_adequacy_ratio)
        + 2.35 * np.maximum(0.0, unprotected_void_pct - 0.35) * transit_vibration_factor * (0.4 + max_fragility)
        + 1.50 * np.maximum(0.0, 0.40 - cushion_adequacy_ratio) / 0.40
        + 3.80 * overload * (0.60 + low_ect_factor) * (distance_multiplier / 2.0)
        + 0.25 * np.tanh(total_weight_kg / 100.0)
        + 0.16 * (handling_nodes - 2)
        + 0.65 * transit_vibration_factor * max_fragility
        + 0.35 * has_liquid
        + 0.20 * orientation_sensitive * (handling_nodes / 5.0)
        + 4.40 * incompatibility_hazard
    )
    prob_true = 1.0 / (1.0 + np.exp(-logit))
    damaged = rng.binomial(1, prob_true)

    df = pd.DataFrame(
        {
            "max_fragility": np.round(max_fragility, 4),
            "mean_fragility": np.round(mean_fragility, 4),
            "unprotected_void_pct": np.round(unprotected_void_pct, 4),
            "cushion_adequacy_ratio": np.round(cushion_adequacy_ratio, 4),
            "total_weight_kg": np.round(total_weight_kg, 2),
            "item_count": item_count,
            "has_liquid": has_liquid,
            "orientation_sensitive": orientation_sensitive,
            "distance_multiplier": distance_multiplier,
            "handling_nodes": handling_nodes,
            "weight_to_limit_ratio": np.round(weight_to_limit_ratio, 4),
            "board_ect_kn_m": board_ect_kn_m,
            "transit_vibration_factor": transit_vibration_factor,
            "incompatibility_hazard": incompatibility_hazard,
            "category_encoded": category_encoded,
            "product_category": categories,
            "damaged": damaged,
        }
    )
    return df


if __name__ == "__main__":
    import os

    frame = generate_dataset(n_samples=12_000)
    out = os.path.join("backend", "ml", "data", "damage_dataset.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Generated {len(frame)} samples -> {out}")
    print(f"Damaged rate: {frame['damaged'].mean():.2%}")
