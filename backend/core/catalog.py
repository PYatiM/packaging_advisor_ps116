"""Unified domain catalog and packaging engineering standards.

Single source of truth grounded in BIS IS 2771 (Part 1): 2022 corrugated
fibreboard standards, ISTA 3A parcel cushioning guidelines, and Indian
cold-chain packaging specifications.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class CategorySpec:
    """Metadata and physical defaults for a product category."""

    code: int
    name: str
    fragility_range: Tuple[float, float]
    base_fragility: float
    default_value_per_kg: float
    esd_sensitive: bool = False
    requires_cold_chain: bool = False
    heavy_duty: bool = False


PRODUCT_CATEGORIES: Dict[str, CategorySpec] = {
    "General": CategorySpec(
        code=0,
        name="General",
        fragility_range=(0.10, 0.35),
        base_fragility=0.25,
        default_value_per_kg=600.0,
    ),
    "Precision Electronics": CategorySpec(
        code=1,
        name="Precision Electronics",
        fragility_range=(0.60, 0.85),
        base_fragility=0.75,
        default_value_per_kg=8000.0,
        esd_sensitive=True,
    ),
    "Perishable Goods": CategorySpec(
        code=2,
        name="Perishable Goods",
        fragility_range=(0.40, 0.70),
        base_fragility=0.60,
        default_value_per_kg=900.0,
        requires_cold_chain=True,
    ),
    "Fine Art & Antiques": CategorySpec(
        code=3,
        name="Fine Art & Antiques",
        fragility_range=(0.80, 0.98),
        base_fragility=0.90,
        default_value_per_kg=12000.0,
    ),
    "Glass & Ceramics": CategorySpec(
        code=4,
        name="Glass & Ceramics",
        fragility_range=(0.75, 0.95),
        base_fragility=0.85,
        default_value_per_kg=1400.0,
    ),
    "Industrial Machinery": CategorySpec(
        code=5,
        name="Industrial Machinery",
        fragility_range=(0.15, 0.40),
        base_fragility=0.25,
        default_value_per_kg=1500.0,
        heavy_duty=True,
    ),
    "Textiles & Apparel": CategorySpec(
        code=6,
        name="Textiles & Apparel",
        fragility_range=(0.05, 0.20),
        base_fragility=0.12,
        default_value_per_kg=950.0,
    ),
    "Pharmaceuticals": CategorySpec(
        code=7,
        name="Pharmaceuticals",
        fragility_range=(0.35, 0.65),
        base_fragility=0.50,
        default_value_per_kg=3500.0,
    ),
    "Automotive Parts": CategorySpec(
        code=8,
        name="Automotive Parts",
        fragility_range=(0.20, 0.50),
        base_fragility=0.35,
        default_value_per_kg=1200.0,
        heavy_duty=True,
    ),
    "Books & Media": CategorySpec(
        code=9,
        name="Books & Media",
        fragility_range=(0.10, 0.25),
        base_fragility=0.18,
        default_value_per_kg=500.0,
    ),
}

CATEGORY_NAMES: List[str] = list(PRODUCT_CATEGORIES.keys())
CATEGORY_MAP: Dict[str, int] = {name: spec.code for name, spec in PRODUCT_CATEGORIES.items()}
FRAGILITY_RANGES: Dict[str, Tuple[float, float]] = {
    name: spec.fragility_range for name, spec in PRODUCT_CATEGORIES.items()
}


@dataclass(frozen=True)
class ContainerSpec:
    """Physical, structural (IS 2771:2022), and commercial container spec."""

    container_id: str
    name: str
    tier_class: str  # "parcel", "cold_parcel", "ptl", "reefer_ptl", "ftl"
    inner_dims_cm: Tuple[float, float, float]
    outer_dims_cm: Tuple[float, float, float]
    tare_weight_kg: float
    max_payload_kg: float
    ply_rating: str
    burst_strength_kgf_cm2: float
    ect_kn_m: float
    material_cost_inr: float
    is_cold_chain: bool = False
    shipping_flat: bool = False
    flat_lane_rate_inr: float = 0.0

    @property
    def inner_volume_cm3(self) -> float:
        l, w, h = self.inner_dims_cm
        return float(l * w * h)

    @property
    def outer_volume_cm3(self) -> float:
        l, w, h = self.outer_dims_cm
        return float(l * w * h)


ORDERED_CONTAINERS: List[ContainerSpec] = [
    ContainerSpec(
        container_id="C-SMALL-01",
        name="3-Ply Corrugated Mailer",
        tier_class="parcel",
        inner_dims_cm=(20.0, 15.0, 12.0),
        outer_dims_cm=(21.0, 16.0, 13.0),
        tare_weight_kg=0.18,
        max_payload_kg=3.0,
        ply_rating="3-Ply",
        burst_strength_kgf_cm2=6.0,
        ect_kn_m=3.2,
        material_cost_inr=18.0,
    ),
    ContainerSpec(
        container_id="C-SLIM-02",
        name="3-Ply Slim Apparel & Book Mailer",
        tier_class="parcel",
        inner_dims_cm=(28.0, 22.0, 10.0),
        outer_dims_cm=(29.0, 23.0, 11.0),
        tare_weight_kg=0.26,
        max_payload_kg=5.0,
        ply_rating="3-Ply",
        burst_strength_kgf_cm2=7.0,
        ect_kn_m=3.6,
        material_cost_inr=26.0,
    ),
    ContainerSpec(
        container_id="CC-EPS-S",
        name="Passive Cold-Chain EPS + Gel Pack (Small)",
        tier_class="cold_parcel",
        inner_dims_cm=(26.0, 20.0, 18.0),
        outer_dims_cm=(31.0, 25.0, 23.0),
        tare_weight_kg=1.40,
        max_payload_kg=5.0,
        ply_rating="5-Ply+EPS",
        burst_strength_kgf_cm2=12.0,
        ect_kn_m=5.0,
        material_cost_inr=135.0,
        is_cold_chain=True,
    ),
    ContainerSpec(
        container_id="C-MED-05",
        name="3-Ply Standard Carton",
        tier_class="parcel",
        inner_dims_cm=(32.0, 25.0, 22.0),
        outer_dims_cm=(33.0, 26.0, 23.0),
        tare_weight_kg=0.42,
        max_payload_kg=8.0,
        ply_rating="3-Ply",
        burst_strength_kgf_cm2=8.0,
        ect_kn_m=4.0,
        material_cost_inr=38.0,
    ),
    ContainerSpec(
        container_id="C-FLAT-04",
        name="5-Ply Flat Laptop & Electronics Carton",
        tier_class="parcel",
        inner_dims_cm=(44.0, 32.0, 15.0),
        outer_dims_cm=(45.5, 33.5, 16.5),
        tare_weight_kg=0.68,
        max_payload_kg=12.0,
        ply_rating="5-Ply",
        burst_strength_kgf_cm2=12.0,
        ect_kn_m=5.2,
        material_cost_inr=68.0,
    ),
    ContainerSpec(
        container_id="CC-EPS-M",
        name="Passive Cold-Chain EPS + PCM (Medium)",
        tier_class="cold_parcel",
        inner_dims_cm=(44.0, 34.0, 30.0),
        outer_dims_cm=(49.0, 39.0, 35.0),
        tare_weight_kg=3.00,
        max_payload_kg=18.0,
        ply_rating="5-Ply+EPS",
        burst_strength_kgf_cm2=14.0,
        ect_kn_m=5.8,
        material_cost_inr=260.0,
        is_cold_chain=True,
    ),
    ContainerSpec(
        container_id="C-LARGE-99",
        name="5-Ply Heavy-Duty Carton",
        tier_class="parcel",
        inner_dims_cm=(50.0, 40.0, 38.0),
        outer_dims_cm=(51.5, 41.5, 39.5),
        tare_weight_kg=1.10,
        max_payload_kg=20.0,
        ply_rating="5-Ply",
        burst_strength_kgf_cm2=12.0,
        ect_kn_m=5.2,
        material_cost_inr=95.0,
    ),
    ContainerSpec(
        container_id="CC-EPS-L",
        name="Passive Cold-Chain EPS + PCM (Large)",
        tier_class="cold_parcel",
        inner_dims_cm=(62.0, 46.0, 42.0),
        outer_dims_cm=(68.0, 52.0, 48.0),
        tare_weight_kg=5.20,
        max_payload_kg=35.0,
        ply_rating="7-Ply+EPS",
        burst_strength_kgf_cm2=18.0,
        ect_kn_m=7.2,
        material_cost_inr=480.0,
        is_cold_chain=True,
    ),
    ContainerSpec(
        container_id="C-HEAVY-5P",
        name="5-Ply Reinforced Carton",
        tier_class="parcel",
        inner_dims_cm=(65.0, 50.0, 45.0),
        outer_dims_cm=(66.5, 51.5, 46.5),
        tare_weight_kg=1.85,
        max_payload_kg=35.0,
        ply_rating="5-Ply",
        burst_strength_kgf_cm2=16.0,
        ect_kn_m=6.8,
        material_cost_inr=165.0,
    ),
    ContainerSpec(
        container_id="C-XL-100",
        name="7-Ply Triple-Wall Jumbo",
        tier_class="parcel",
        inner_dims_cm=(100.0, 100.0, 100.0),
        outer_dims_cm=(102.5, 102.5, 102.5),
        tare_weight_kg=4.50,
        max_payload_kg=65.0,
        ply_rating="7-Ply",
        burst_strength_kgf_cm2=24.0,
        ect_kn_m=9.6,
        material_cost_inr=380.0,
    ),
    ContainerSpec(
        container_id="PALLET-STD",
        name="ISPM-15 Euro/Indian Pallet",
        tier_class="ptl",
        inner_dims_cm=(120.0, 100.0, 150.0),
        outer_dims_cm=(120.0, 100.0, 165.0),
        tare_weight_kg=20.0,
        max_payload_kg=1000.0,
        ply_rating="Wood/7-Ply",
        burst_strength_kgf_cm2=35.0,
        ect_kn_m=12.4,
        material_cost_inr=650.0,
        is_cold_chain=False,
        shipping_flat=False,
    ),
    ContainerSpec(
        container_id="PTL-REEFER",
        name="Active Refrigerated PTL Unit",
        tier_class="reefer_ptl",
        inner_dims_cm=(240.0, 120.0, 180.0),
        outer_dims_cm=(240.0, 120.0, 195.0),
        tare_weight_kg=45.0,
        max_payload_kg=2500.0,
        ply_rating="Insulated Reefer",
        burst_strength_kgf_cm2=40.0,
        ect_kn_m=15.0,
        material_cost_inr=1200.0,
        is_cold_chain=True,
        shipping_flat=True,
        flat_lane_rate_inr=3800.0,
    ),
    ContainerSpec(
        container_id="TRUCK-20FT",
        name="20-ft Closed Container FTL",
        tier_class="ftl",
        inner_dims_cm=(590.0, 235.0, 239.0),
        outer_dims_cm=(606.0, 244.0, 259.0),
        tare_weight_kg=0.0,
        max_payload_kg=15000.0,
        ply_rating="Steel FTL",
        burst_strength_kgf_cm2=50.0,
        ect_kn_m=15.0,
        material_cost_inr=3500.0,
        is_cold_chain=False,
        shipping_flat=True,
        flat_lane_rate_inr=18000.0,
    ),
    ContainerSpec(
        container_id="TRUCK-40FT-HC",
        name="40-ft High-Cube FTL",
        tier_class="ftl",
        inner_dims_cm=(1203.0, 235.0, 270.0),
        outer_dims_cm=(1219.0, 244.0, 290.0),
        tare_weight_kg=0.0,
        max_payload_kg=28000.0,
        ply_rating="Steel FTL",
        burst_strength_kgf_cm2=50.0,
        ect_kn_m=15.0,
        material_cost_inr=5500.0,
        is_cold_chain=False,
        shipping_flat=True,
        flat_lane_rate_inr=28000.0,
    ),
]

CONTAINER_CATALOG: Dict[str, ContainerSpec] = {
    spec.container_id: spec for spec in ORDERED_CONTAINERS
}


def calculate_item_fragility(
    category: str,
    length_cm: float,
    width_cm: float,
    height_cm: float,
    weight_kg: float,
) -> float:
    """Compute item fragility score in [0.05, 0.98] from category, density, and slenderness."""
    spec = PRODUCT_CATEGORIES.get(category, PRODUCT_CATEGORIES["General"])
    base_frag = spec.base_fragility

    vol_cm3 = max(1e-6, length_cm * width_cm * height_cm)
    density = max(0.0, weight_kg) / (vol_cm3 / 1000.0)
    if density > 0.0:
        base_frag -= math.log10(density + 1.0) * 0.04

    dims = sorted([max(0.1, length_cm), max(0.1, width_cm), max(0.1, height_cm)])
    ratio = dims[2] / dims[0]
    if ratio > 1.0:
        base_frag += math.log(ratio) * 0.045

    return max(0.05, min(0.98, round(base_frag, 2)))


def estimate_item_value_inr(
    category: str,
    weight_kg: float,
    declared_value_inr: Optional[float] = None,
) -> float:
    """Return per-unit cargo value in INR with diminishing bulk weight scaling."""
    if declared_value_inr is not None and declared_value_inr > 0.0:
        return float(declared_value_inr)
    spec = PRODUCT_CATEGORIES.get(category, PRODUCT_CATEGORIES["General"])
    wt = float(weight_kg)
    eff_wt = wt if wt <= 15.0 else 15.0 + ((wt - 15.0) ** 0.65) * 2.5
    return max(150.0, round(eff_wt * spec.default_value_per_kg, 2))


def compute_cushion_pad_cm(fragility_score: float, weight_kg: float) -> float:
    """Return ISTA 3A per-face protective cushioning clearance thickness in cm."""
    if fragility_score < 0.25:
        return 0.0
    if fragility_score < 0.50:
        return 0.5
    if fragility_score < 0.75:
        return 2.0 if weight_kg >= 5.0 else 1.5
    return 3.0 if weight_kg >= 5.0 else 2.5


def compute_incompatibility_hazard(
    items: list,
    is_cold_chain_container: bool = True,
    transit_days_min: int = 1,
) -> float:
    """Score [0.0, 1.0] hazard from thermal exposure, liquid leaks, and heavy-vs-fragile mixing."""
    if not items:
        return 0.0

    hazard = 0.0
    has_cold_chain_need = any(
        getattr(i, "product_category", "") == "Perishable Goods"
        or getattr(i, "requires_cold_chain", False)
        for i in items
    )
    if has_cold_chain_need and (not is_cold_chain_container and transit_days_min > 1):
        hazard += 0.50

    vulnerable_cats = {
        "Precision Electronics",
        "Fine Art & Antiques",
        "Books & Media",
        "Textiles & Apparel",
        "Pharmaceuticals",
    }
    has_liquid = any(getattr(i, "is_liquid", False) for i in items)
    if has_liquid:
        has_vulnerable_non_liquid = any(
            getattr(i, "product_category", "General") in vulnerable_cats
            and not getattr(i, "is_liquid", False)
            for i in items
        )
        if has_vulnerable_non_liquid:
            hazard += 0.35

    fragile_items = [i for i in items if getattr(i, "fragility_score", 0.0) >= 0.70]
    heavy_items = [
        i
        for i in items
        if (getattr(i, "weight_kg", 0.0) >= 5.0 and getattr(i, "fragility_score", 0.0) <= 0.40)
        or getattr(i, "product_category", "") in ("Industrial Machinery", "Automotive Parts")
    ]
    if fragile_items and heavy_items:
        if any(a is not b for a in fragile_items for b in heavy_items):
            hazard += 0.35

    return max(0.0, min(1.0, round(hazard, 4)))
