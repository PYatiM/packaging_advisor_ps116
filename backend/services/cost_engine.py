"""Dynamic Indian parcel, cold-chain, PTL, and FTL freight cost engine.

Computes tare-adjusted gross weight, dimensional weight (cm^3/5000 for
parcels, stacked-height 7 kg/CFT density billing for PTL), slab rounding,
10 kg+ tapering, realistic void-fill dunnage costs, FTL zone multipliers,
Scope 3 CO2e emissions, and risk-adjusted effective landed cost.
"""

import math
from typing import Dict, List, Optional, Tuple

from backend.core.catalog import (
    CONTAINER_CATALOG,
    ORDERED_CONTAINERS,
    ContainerSpec,
    estimate_item_value_inr,
)
from backend.core.config import settings
from backend.services.routing_engine import RouteProfile


class DynamicCostEngine:
    """Calculates material, dunnage, freight, CO2e, and risk-adjusted landed costs."""

    _MODE_EMISSION_FACTORS = {
        "Air": 0.602,
        "Last-Mile": 0.160,
        "Refrigerated Road": 0.135,
        "Priority Cold Surface": 0.105,
        "Surface": 0.085,
        "PTL Surface": 0.068,
    }

    _FTL_ZONE_MULT = {
        "A": 0.50,
        "B": 0.95,
        "C": 2.10,
        "D": 2.45,
        "E": 3.10,
    }

    def get_current_rates(self) -> Dict[str, dict]:
        """Return container material and flat-lane rates from the unified catalog."""
        rates: Dict[str, dict] = {}
        for spec in ORDERED_CONTAINERS:
            entry = {
                "material": spec.material_cost_inr,
                "shipping_flat": spec.shipping_flat,
            }
            if spec.shipping_flat:
                entry["flat_rate"] = spec.flat_lane_rate_inr
            rates[spec.container_id] = entry
        return rates

    @classmethod
    def compute_co2_emissions_kg(
        cls,
        gross_wt_kg: float,
        tare_wt_kg: float,
        distance_km: float,
        mode_name: str,
        is_cold_parcel: bool = False,
    ) -> float:
        """Estimate Scope 3 transport + embodied container CO2e emissions (kg)."""
        factor = cls._MODE_EMISSION_FACTORS.get(mode_name, 0.085)
        transport_co2 = (max(0.0, gross_wt_kg) / 1000.0) * max(1.0, distance_km) * factor
        packaging_co2 = max(0.0, tare_wt_kg) * (1.85 if is_cold_parcel else 0.92)
        return round(transport_co2 + packaging_co2, 3)

    @staticmethod
    def round_billable_weight(weight_kg: float) -> float:
        """Apply courier slab rounding: 0.5 kg steps up to 5 kg, 1.0 kg above."""
        if weight_kg <= 0.0:
            return 0.0
        if weight_kg <= 5.0:
            return math.ceil(weight_kg * 2.0) / 2.0
        return float(math.ceil(weight_kg))

    @staticmethod
    def compute_dunnage_cost(items: list, candidate: dict) -> float:
        """Estimate void-fill dunnage cost (kraft paper / air pillows / EPE foam)."""
        spec: Optional[ContainerSpec] = candidate.get("spec")
        if spec is not None and spec.tier_class in ("ptl", "reefer_ptl", "ftl"):
            return 0.0
        box_id = candidate.get("box_id", "")
        if "MULTI-TRUCK" in box_id or box_id == "PALLET-STD":
            return 0.0

        padded_util = candidate.get(
            "padded_utilization_pct", candidate.get("utilization_pct", 0.5)
        )
        unprotected_void_pct = max(0.0, 1.0 - padded_util)
        box_vol_cm3 = candidate.get("box_vol", 0.0)
        void_litres = (box_vol_cm3 * unprotected_void_pct) / 1000.0

        max_frag = max((i.fragility_score for i in items), default=0.3)
        if max_frag < 0.30:
            rate = 0.45  # Crushed kraft paper
        elif max_frag < 0.70:
            rate = 0.75  # Multi-cushion air pillows
        else:
            rate = 1.60  # High-resilience EPE foam inserts

        base_material = float(candidate.get("b_cost", 100.0))
        return round(min(void_litres * rate, 0.55 * base_material), 2)

    @staticmethod
    def get_eligible_modes(
        candidate: dict,
        route: RouteProfile,
    ) -> List[Tuple[str, float, str, str]]:
        """Return eligible (mode_name, base_rate, advice, estimated_days) tuples."""
        box_id = candidate.get("box_id", "")
        spec: Optional[ContainerSpec] = candidate.get("spec") or CONTAINER_CATALOG.get(
            box_id
        )
        tier_class = spec.tier_class if spec is not None else "parcel"

        if tier_class == "ftl" or "MULTI-TRUCK" in box_id:
            days = f"{route.surface_days_min}-{route.surface_days_max} days"
            return [
                (
                    "Surface",
                    settings.SURFACE_BASE_RATE_PER_KG,
                    f"Dedicated FTL Road Freight -- {days} ({route.shipping_zone_tier}).",
                    days,
                )
            ]

        if tier_class == "reefer_ptl" or box_id == "PTL-REEFER":
            days = f"{route.surface_days_min}-{route.surface_days_max} days"
            return [
                (
                    "Refrigerated Road",
                    20.0,
                    f"Active Refrigerated Cold-Chain Road -- {days} ({route.shipping_zone_tier}).",
                    days,
                )
            ]

        if tier_class == "cold_parcel" or "INSULATED" in box_id:
            air_days = (
                "1 day"
                if route.air_days_min == route.air_days_max == 1
                else f"{route.air_days_min}-{route.air_days_max} days"
            )
            modes: List[Tuple[str, float, str, str]] = []
            if route.zone_code == "A":
                modes.append(
                    (
                        "Last-Mile",
                        settings.SURFACE_BASE_RATE_PER_KG * 1.05,
                        f"Hyper-Local Cold-Chain Last-Mile -- 1-2 days ({route.shipping_zone_tier}).",
                        "1-2 days",
                    )
                )
            elif route.zone_code == "B" and route.surface_days_max <= 3:
                surf_days = f"{route.surface_days_min}-{route.surface_days_max} days"
                modes.append(
                    (
                        "Priority Cold Surface",
                        settings.SURFACE_BASE_RATE_PER_KG * 1.45,
                        f"Priority Cold Surface (<48h PCM holdover) -- {surf_days} ({route.shipping_zone_tier}).",
                        surf_days,
                    )
                )
            modes.append(
                (
                    "Air",
                    settings.AIR_BASE_RATE_PER_KG * 1.15,
                    f"Express Cold-Chain Air Freight -- {air_days} ({route.shipping_zone_tier}).",
                    air_days,
                )
            )
            return modes

        if tier_class == "ptl" or box_id == "PALLET-STD":
            days = f"{route.surface_days_min}-{route.surface_days_max} days"
            return [
                (
                    "PTL Surface",
                    settings.PTL_BASE_RATE_PER_KG,
                    f"B2B Part-Truckload (PTL) CFT Road Freight -- {days} ({route.shipping_zone_tier}).",
                    days,
                )
            ]

        # Standard parcel cartons & custom crates
        if route.zone_code == "A":
            return [
                (
                    "Last-Mile",
                    settings.SURFACE_BASE_RATE_PER_KG * 0.75,
                    f"Last-Mile Local Road Delivery -- 1-2 days ({route.shipping_zone_tier}).",
                    "1-2 days",
                )
            ]

        surf_days = f"{route.surface_days_min}-{route.surface_days_max} days"
        air_days = (
            f"{route.air_days_min} days"
            if route.air_days_min == route.air_days_max
            else f"{route.air_days_min}-{route.air_days_max} days"
        )
        return [
            (
                "Surface",
                settings.SURFACE_BASE_RATE_PER_KG,
                f"Standard Surface Freight -- {surf_days} ({route.shipping_zone_tier}).",
                surf_days,
            ),
            (
                "Air",
                settings.AIR_BASE_RATE_PER_KG,
                f"Express Air Freight -- {air_days} ({route.shipping_zone_tier}).",
                air_days,
            ),
        ]

    def calculate_mode_pricing(
        self,
        items: list,
        candidate: dict,
        route: RouteProfile,
        mode_name: str,
        base_rate: float,
    ) -> dict:
        """Compute material, dunnage, freight, chargeable weight, CO2e, and direct cost."""
        box_id = candidate["box_id"]
        box_count = max(1, int(candidate.get("box_count", 1)))
        spec: Optional[ContainerSpec] = candidate.get("spec") or CONTAINER_CATALOG.get(
            box_id
        )

        payload_wt = sum(
            i.weight_kg * getattr(i, "quantity", 1) for i in items
        )
        tare_wt = float(candidate.get("tare_wt", 0.0))
        gross_wt = round(payload_wt + tare_wt, 3)

        outer_vol = float(candidate.get("outer_vol", candidate.get("box_vol", 0.0)))
        material_cost = round(float(candidate.get("b_cost", 0.0)), 2)
        dunnage_cost = self.compute_dunnage_cost(items, candidate)

        if "MULTI-TRUCK" in box_id:
            try:
                trucks = int(box_id.split("(")[1].split("x")[0])
            except (IndexError, ValueError):
                trucks = 2
            material_cost = round(trucks * 5500.0, 2)
            flat_rate = trucks * 28000.0
            lane_mult = self._FTL_ZONE_MULT.get(
                route.zone_code, route.distance_multiplier
            )
            freight_cost = round(flat_rate * lane_mult, 2)
            vol_wt = outer_vol / settings.VOLUMETRIC_DIVISOR_PARCEL
            charge_wt = round(max(gross_wt, vol_wt), 2)
        elif spec is not None and spec.shipping_flat:
            lane_mult = self._FTL_ZONE_MULT.get(
                route.zone_code, route.distance_multiplier
            )
            freight_cost = round(spec.flat_lane_rate_inr * lane_mult, 2)
            vol_wt = outer_vol / settings.VOLUMETRIC_DIVISOR_PARCEL
            charge_wt = round(max(gross_wt, vol_wt), 2)
        elif spec is not None and spec.tier_class == "ptl":
            placements = candidate.get("placements") or []
            if placements:
                max_z = max(float(p["z_cm"]) + float(p["dz_cm"]) for p in placements)
            else:
                max_z = (
                    float(candidate.get("box_vol", 0.0))
                    * max(0.20, float(candidate.get("padded_utilization_pct", 0.35)))
                    / (120.0 * 100.0)
                )
            stacked_h_cm = min(165.0, max(45.0, max_z + 15.0))
            eff_outer_vol = 120.0 * 100.0 * stacked_h_cm * box_count
            cft = eff_outer_vol / 28316.85
            vol_wt = cft * settings.PTL_CFT_KG_FACTOR
            charge_wt = self.round_billable_weight(max(50.0, gross_wt, vol_wt))
            freight_cost = 250.0 + (
                charge_wt * settings.PTL_BASE_RATE_PER_KG * route.distance_multiplier
            )
            if route.is_oda_special:
                freight_cost += 150.0
            freight_cost = round(freight_cost, 2)
        else:
            vol_wt = outer_vol / settings.VOLUMETRIC_DIVISOR_PARCEL
            charge_wt = self.round_billable_weight(max(gross_wt, vol_wt))
            tapered_wt = min(charge_wt, 10.0) + max(0.0, charge_wt - 10.0) * 0.80

            if mode_name == "Air":
                docket_fee = 50.0 * box_count
            elif mode_name == "Last-Mile":
                docket_fee = 25.0 * box_count
            else:
                docket_fee = 35.0 * box_count

            freight_cost = docket_fee + (
                tapered_wt * base_rate * route.distance_multiplier
            )
            if route.is_oda_special:
                freight_cost += 60.0 * box_count
            freight_cost = round(freight_cost, 2)

        direct_cost = round(material_cost + dunnage_cost + freight_cost, 2)
        declared_value = round(
            sum(
                estimate_item_value_inr(
                    i.product_category,
                    i.weight_kg,
                    getattr(i, "declared_value_inr", None),
                )
                * getattr(i, "quantity", 1)
                for i in items
            ),
            2,
        )

        is_cold_parcel = spec is not None and spec.tier_class == "cold_parcel"
        co2_kg = self.compute_co2_emissions_kg(
            gross_wt,
            tare_wt,
            getattr(route, "estimated_distance_km", 1000.0),
            mode_name,
            is_cold_parcel=is_cold_parcel,
        )

        return {
            "material_cost": material_cost,
            "dunnage_cost": dunnage_cost,
            "freight_cost": freight_cost,
            "direct_cost": direct_cost,
            "gross_wt_kg": round(gross_wt, 2),
            "vol_wt_kg": round(vol_wt, 2),
            "chargeable_wt_kg": round(charge_wt, 2),
            "declared_value_inr": declared_value,
            "co2_kg": co2_kg,
        }

    @staticmethod
    def sort_options(options: List[dict], priority: str = "balanced") -> List[dict]:
        """Rank (container x mode) candidates according to optimization priority."""
        if priority == "lowest_cost":
            return sorted(options, key=lambda x: (x["t_cost"], x["r_prob"]))
        if priority == "lowest_risk":
            return sorted(options, key=lambda x: (x["r_prob"], x["t_cost"]))
        if priority == "eco_friendly":
            return sorted(
                options,
                key=lambda x: (
                    x["effective_landed_cost"] + x.get("co2_kg", 0.0) * 45.0,
                    x.get("co2_kg", 0.0),
                    x["t_cost"],
                ),
            )
        return sorted(
            options,
            key=lambda x: (x["effective_landed_cost"], x["t_cost"]),
        )
