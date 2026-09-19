"""
DamagePredictor — ML-backed transit damage-risk scorer
======================================================

Loads the trained RandomForest/XGBoost model persisted by
``train_damage_model.py`` and returns calibrated damage-risk
probabilities for a given set of items and box utilisation.

Falls back to the original heuristic formula if the model artifact
is missing (e.g. first deployment before training has been run).
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Feature columns expected by the trained model (must match training order)
_FEATURE_COLS = [
    "max_fragility",
    "void_space_pct",
    "total_weight_kg",
    "item_count",
    "has_liquid",
    "orientation_sensitive",
    "distance_multiplier",
    "weight_density",
    "category_encoded",
]

# Category encoding map — mirrors dataset_generator.py
_CATEGORY_MAP = {
    "General":                0,
    "Precision Electronics":  1,
    "Perishable Goods":       2,
    "Fine Art & Antiques":    3,
    "Glass & Ceramics":       4,
    "Industrial Machinery":   5,
    "Textiles & Apparel":     6,
    "Pharmaceuticals":        7,
}

_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_MODEL_PATH = os.path.join(_ARTIFACT_DIR, "best_damage_model.joblib")
_SCALER_PATH = os.path.join(_ARTIFACT_DIR, "feature_scaler.joblib")


class DamagePredictor:
    """Predict transit-damage probability for a group of items.

    On construction the class tries to load the trained model and
    feature scaler from ``backend/ml/artifacts/``.  If the files are
    not found, all calls to :meth:`predict_damage_risk` silently fall
    back to the original rule-based heuristic so the application keeps
    working even without a training run.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False

        try:
            import joblib
            if os.path.exists(_MODEL_PATH) and os.path.exists(_SCALER_PATH):
                self.model = joblib.load(_MODEL_PATH)
                self.scaler = joblib.load(_SCALER_PATH)
                self.model_loaded = True
                logger.info("DamagePredictor: loaded trained model from %s", _MODEL_PATH)
            else:
                logger.warning(
                    "DamagePredictor: model artifacts not found — "
                    "falling back to heuristic.  Run "
                    "'python -m backend.ml.train_damage_model' to train."
                )
        except Exception as exc:
            logger.warning("DamagePredictor: failed to load model (%s) — using heuristic.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_damage_risk(
        self,
        items: list,
        utilization_pct: float,
        distance_multiplier: float = 1.0,
    ) -> float:
        """Return a damage probability in [0.01, 0.99].

        Parameters
        ----------
        items : list[SKUItem]
            Pydantic SKU items from the cart schema.
        utilization_pct : float
            Box volume utilisation (0-1).
        distance_multiplier : float, optional
            Zone-to-zone distance factor (default 1.0).
        """
        if not items:
            return 0.0

        if self.model_loaded:
            return self._predict_ml(items, utilization_pct, distance_multiplier)
        return self._predict_heuristic(items, utilization_pct)

    # ------------------------------------------------------------------
    # ML-based prediction
    # ------------------------------------------------------------------
    def _predict_ml(self, items, utilization_pct, distance_multiplier):
        features = self._extract_features(items, utilization_pct, distance_multiplier)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0, 1]
        return max(0.01, min(float(prob), 0.99))

    def _extract_features(self, items, utilization_pct, distance_multiplier):
        """Build the 9-element feature vector expected by the model."""
        max_fragility = max(item.fragility_score for item in items)
        void_space_pct = max(0.0, 1.0 - utilization_pct)
        total_weight = sum(item.weight_kg for item in items)
        item_count = len(items)
        has_liquid = int(any(item.is_liquid for item in items))
        orientation_sensitive = int(any(item.orientation_sensitive for item in items))

        # Approximate box volume and weight density
        item_vol = sum(
            item.length_cm * item.width_cm * item.height_cm
            for item in items
        )
        box_vol = item_vol / (1.0 - void_space_pct + 1e-9) if void_space_pct < 1.0 else item_vol
        weight_density = total_weight / (box_vol / 1000.0 + 1e-9)  # kg per litre

        # Category: use the most fragile item's category as representative
        most_fragile = max(items, key=lambda i: i.fragility_score)
        cat = getattr(most_fragile, "product_category", "General")
        category_encoded = _CATEGORY_MAP.get(cat, 0)

        return [
            max_fragility,
            void_space_pct,
            total_weight,
            item_count,
            has_liquid,
            orientation_sensitive,
            distance_multiplier,
            weight_density,
            category_encoded,
        ]

    # ------------------------------------------------------------------
    # Heuristic fallback (original formula, preserved for safety)
    # ------------------------------------------------------------------
    @staticmethod
    def _predict_heuristic(items, utilization_pct):
        """Original rule-based risk estimate (no ML model required)."""
        max_fragility = max(item.fragility_score for item in items)
        total_weight = sum(item.weight_kg for item in items)
        void_space_pct = max(0.0, 1.0 - utilization_pct)
        weight_risk = (total_weight / (total_weight + 20.0)) * 0.3
        risk = (max_fragility * 0.4) + (void_space_pct * 0.3) + weight_risk
        return max(0.01, min(risk, 0.99))
