class DamagePredictor:
    def __init__(self):
        # In a real environment, this would load an MLflow XGBoost model
        self.model_loaded = True

    def predict_damage_risk(self, items: list, utilization_pct: float) -> float:
        """
        Calculates transit risk based on the exact volumetric void space and item fragility.
        High void space exponentially increases shifting and shattering risk.
        """
        if not items:
            return 0.0
            
        max_fragility = max([item.fragility_score for item in items])
        total_weight = sum([item.weight_kg for item in items])
        
        # True void space percentage (e.g. 1.0 - 0.75 = 0.25)
        void_space_pct = max(0.0, 1.0 - utilization_pct)
        
        # Use an asymptotic function for weight to prevent unbounded growth
        weight_risk = (total_weight / (total_weight + 20.0)) * 0.3
        
        # Risk scales mathematically with empty space, fragility, and weight
        risk = (max_fragility * 0.4) + (void_space_pct * 0.3) + weight_risk
        return max(0.01, min(risk, 0.99))
