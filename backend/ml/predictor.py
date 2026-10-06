"""DamagePredictor — calibrated ML transit damage-risk scorer.

Loads the calibrated classifier and feature scaler persisted by
``train_damage_model.py`` and predicts transit-damage probabilities in
[0.01, 0.99] using 15 physics-grounded container, cushioning, route,
mode-vibration, and cargo-compatibility features.
"""

import logging
import math
import os
from typing import Optional
import numpy as np

from backend.core.catalog import (
    CATEGORY_MAP,
    ContainerSpec,
    compute_incompatibility_hazard,
)

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_MODEL_PATH = os.path.join(_ARTIFACT_DIR, "best_damage_model.joblib")
_SCALER_PATH = os.path.join(_ARTIFACT_DIR, "feature_scaler.joblib")

_MODE_VIBRATION = {
    "Air": 0.45,
    "Last-Mile": 0.55,
    "Refrigerated Road": 0.65,
    "Priority Cold Surface": 0.75,
    "PTL Surface": 0.90,
    "Surface": 1.00,
}

_PREDICTOR_SINGLETON: Optional["DamagePredictor"] = None


class DamagePredictor:
    """Predict calibrated transit-damage probability for a shipment group."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False

        try:
            import joblib

            if os.path.exists(_MODEL_PATH) and os.path.exists(_SCALER_PATH):
                loaded_model = joblib.load(_MODEL_PATH)
                loaded_scaler = joblib.load(_SCALER_PATH)
                expected_features = getattr(loaded_scaler, "n_features_in_", 15)
                if expected_features == 15:
                    for cc in getattr(loaded_model, "calibrated_classifiers_", []):
                        est = getattr(cc, "estimator", None)
                        if est is not None and hasattr(est, "n_jobs"):
                            est.n_jobs = 1
                    self.model = loaded_model
                    self.scaler = loaded_scaler
                    self.model_loaded = True
                    self._pred_cache = {}
                    logger.info(
                        "DamagePredictor: loaded calibrated 15-feature model from %s",
                        _MODEL_PATH,
                    )
                else:
                    logger.warning(
                        "DamagePredictor: scaler expects %s features (need 15) — "
                        "using physics heuristic until retrained.",
                        expected_features,
                    )
            else:
                logger.warning(
                    "DamagePredictor: model artifacts not found — using heuristic."
                )
        except Exception as exc:
            logger.warning(
                "DamagePredictor: failed to load model (%s) — using heuristic.", exc
            )

    @staticmethod
    def _infer_handling_nodes(distance_multiplier: float) -> int:
        if distance_multiplier <= 1.05:
            return 2
        if distance_multiplier <= 1.40:
            return 3
        if distance_multiplier <= 1.80:
            return 4
        if distance_multiplier <= 2.50:
            return 5
        return 7

    def _extract_features(
        self,
        items: list,
        utilization_pct: float,
        distance_multiplier: float,
        *,
        padded_utilization_pct: Optional[float] = None,
        cushion_adequacy_ratio: Optional[float] = None,
        container_spec: Optional[ContainerSpec] = None,
        transit_mode: str = "Surface",
        handling_nodes: Optional[int] = None,
        transit_days_min: int = 1,
        box_count: int = 1,
    ) -> list:
        """Build the 15-element feature vector matching dataset_generator.py."""
        quantities = [max(1, int(getattr(i, "quantity", 1))) for i in items]
        total_qty = sum(quantities)

        max_fragility = max(i.fragility_score for i in items)
        mean_fragility = (
            sum(i.fragility_score * q for i, q in zip(items, quantities)) / total_qty
        )

        eff_padded_util = (
            padded_utilization_pct
            if padded_utilization_pct is not None
            else utilization_pct
        )
        unprotected_void_pct = max(0.0, min(1.0, 1.0 - eff_padded_util))
        cushion_ratio = (
            float(cushion_adequacy_ratio)
            if cushion_adequacy_ratio is not None
            else 1.0
        )

        total_weight_kg = sum(
            i.weight_kg * q for i, q in zip(items, quantities)
        )
        has_liquid = int(any(i.is_liquid for i in items))
        orientation_sensitive = int(any(i.orientation_sensitive for i in items))

        nodes = (
            int(handling_nodes)
            if handling_nodes is not None
            else self._infer_handling_nodes(distance_multiplier)
        )

        if container_spec is not None:
            board_ect = container_spec.ect_kn_m
            max_cap = container_spec.max_payload_kg * max(1, box_count)
            is_cold_container = container_spec.is_cold_chain
        else:
            board_ect = 5.2
            max_cap = 25.0 * max(1, box_count)
            is_cold_container = True

        weight_to_limit_ratio = min(1.5, total_weight_kg / max(1.0, max_cap))
        vibration = _MODE_VIBRATION.get(transit_mode, 1.00)
        incompat_hazard = compute_incompatibility_hazard(
            items,
            is_cold_chain_container=is_cold_container,
            transit_days_min=transit_days_min,
        )

        most_fragile = max(items, key=lambda i: i.fragility_score)
        cat = getattr(most_fragile, "product_category", "General")
        category_encoded = CATEGORY_MAP.get(cat, 0)

        return [
            max_fragility,
            mean_fragility,
            unprotected_void_pct,
            cushion_ratio,
            total_weight_kg,
            total_qty,
            has_liquid,
            orientation_sensitive,
            distance_multiplier,
            nodes,
            weight_to_limit_ratio,
            board_ect,
            vibration,
            incompat_hazard,
            category_encoded,
        ]

    def predict_damage_risk(
        self,
        items: list,
        utilization_pct: float,
        distance_multiplier: float = 1.0,
        *,
        padded_utilization_pct: Optional[float] = None,
        cushion_adequacy_ratio: Optional[float] = None,
        container_spec: Optional[ContainerSpec] = None,
        transit_mode: str = "Surface",
        handling_nodes: Optional[int] = None,
        transit_days_min: int = 1,
        box_count: int = 1,
    ) -> float:
        """Return calibrated transit-damage probability in [0.01, 0.99]."""
        if not items:
            return 0.0

        features = self._extract_features(
            items,
            utilization_pct,
            distance_multiplier,
            padded_utilization_pct=padded_utilization_pct,
            cushion_adequacy_ratio=cushion_adequacy_ratio,
            container_spec=container_spec,
            transit_mode=transit_mode,
            handling_nodes=handling_nodes,
            transit_days_min=transit_days_min,
            box_count=box_count,
        )

        if self.model_loaded:
            key = tuple(round(float(v), 4) for v in features)
            cached = self._pred_cache.get(key)
            if cached is not None:
                return cached
            X = np.array([features], dtype=float)
            X_scaled = self.scaler.transform(X)
            prob = float(self.model.predict_proba(X_scaled)[0, 1])
            val = round(max(0.01, min(prob, 0.99)), 4)
            if len(self._pred_cache) < 8192:
                self._pred_cache[key] = val
            return val

        return self._predict_heuristic_from_features(features)

    @staticmethod
    def _predict_heuristic_from_features(features: list) -> float:
        """Physics-grounded fallback heuristic matching logistic priors."""
        (
            max_frag,
            mean_frag,
            void_pct,
            cushion_ratio,
            total_wt,
            _item_count,
            has_liquid,
            orient_sens,
            dist_mult,
            nodes,
            wt_ratio,
            ect,
            vib,
            hazard,
            _cat_enc,
        ) = features

        overload = max(0.0, wt_ratio - 0.85)
        low_ect = max(0.0, (6.0 - ect) / 3.0)
        logit = (
            -4.45
            + 1.35 * max_frag
            + 0.40 * mean_frag
            + 3.60 * max_frag * max(0.0, 0.85 - cushion_ratio)
            + 2.35 * max(0.0, void_pct - 0.35) * vib * (0.4 + max_frag)
            + 1.50 * max(0.0, 0.40 - cushion_ratio) / 0.40
            + 3.80 * overload * (0.60 + low_ect) * (dist_mult / 2.0)
            + 0.25 * math.tanh(total_wt / 100.0)
            + 0.16 * (nodes - 2)
            + 0.65 * vib * max_frag
            + 0.35 * has_liquid
            + 0.20 * orient_sens * (nodes / 5.0)
            + 4.40 * hazard
        )
        prob = 1.0 / (1.0 + math.exp(-logit))
        return round(max(0.01, min(prob, 0.99)), 4)


def get_damage_predictor() -> DamagePredictor:
    """Return a cached singleton DamagePredictor instance."""
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is None:
        _PREDICTOR_SINGLETON = DamagePredictor()
    return _PREDICTOR_SINGLETON
