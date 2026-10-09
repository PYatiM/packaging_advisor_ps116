"""Smart cargo compatibility grouping for multi-SKU carts.

Segregates cargo that cannot safely share a single container (cold-chain
perishables, liquids, ESD-sensitive electronics, delicate fragile goods vs.
heavy rigid parts) while consolidating compatible dry cargo to avoid
unnecessary multi-docket shipping fees.
"""

from typing import List, Tuple
from backend.schemas.cart import SKUItem


_FRAGILE_CATEGORIES = {"Fine Art & Antiques", "Glass & Ceramics"}
_HEAVY_CATEGORIES = {"Industrial Machinery", "Automotive Parts"}


def _is_fragile_dry(item: SKUItem) -> bool:
    return (
        item.product_category in _FRAGILE_CATEGORIES
        or item.fragility_score >= 0.70
    )


def _is_heavy_rigid(item: SKUItem) -> bool:
    return (
        item.product_category in _HEAVY_CATEGORIES
        or item.weight_kg >= 8.0
    )


def group_cart_items(
    items: List[SKUItem],
) -> Tuple[List[Tuple[str, List[SKUItem]]], List[str]]:
    """Split cart items into compatible shipment groups with reasons."""
    if not items:
        return [], []

    perishables: List[SKUItem] = []
    liquids: List[SKUItem] = []
    electronics: List[SKUItem] = []
    dry_remaining: List[SKUItem] = []

    for item in items:
        if (
            item.product_category == "Perishable Goods"
            or getattr(item, "requires_cold_chain", False)
        ):
            perishables.append(item)
        elif item.is_liquid:
            liquids.append(item)
        elif item.product_category == "Precision Electronics":
            electronics.append(item)
        else:
            dry_remaining.append(item)

    fragile_dry = [i for i in dry_remaining if _is_fragile_dry(i)]
    non_fragile_dry = [i for i in dry_remaining if not _is_fragile_dry(i)]

    heavy_dry = [i for i in non_fragile_dry if _is_heavy_rigid(i)]
    light_dry = [i for i in non_fragile_dry if not _is_heavy_rigid(i)]

    groups: List[Tuple[str, List[SKUItem]]] = []
    reasons: List[str] = []

    if perishables:
        groups.append(("perishables", perishables))
    if liquids:
        groups.append(("liquids", liquids))
    if electronics:
        groups.append(("electronics", electronics))

    if fragile_dry and non_fragile_dry:
        groups.append(("fragile_delicate", fragile_dry))
        reasons.append(
            "Isolated delicate fragile cargo (glass, ceramics, fine art) from heavier or rigid dry goods to prevent impact crushing."
        )
    elif fragile_dry:
        groups.append(("fragile_delicate", fragile_dry))

    heavy_total_wt = sum(
        i.weight_kg * getattr(i, "quantity", 1) for i in heavy_dry
    )
    if heavy_dry and light_dry and heavy_total_wt >= 10.0:
        groups.append(("heavy_industrial", heavy_dry))
        groups.append(("general", light_dry))
        reasons.append(
            f"Segregated heavy rigid parts ({heavy_total_wt:.1f} kg) from lightweight general cargo to prevent crush deformation."
        )
    elif non_fragile_dry:
        groups.append(("general", non_fragile_dry))

    if len(groups) > 1:
        active_labels = {lbl for lbl, _ in groups}
        if "perishables" in active_labels:
            reasons.insert(
                0,
                "Separated perishable / cold-chain items into dedicated insulated packaging.",
            )
        if "liquids" in active_labels:
            reasons.append(
                "Isolated liquid SKUs to prevent leak contamination across electronics, textiles, or paper goods."
            )
        if "electronics" in active_labels:
            reasons.append(
                "Separated precision electronics into ESD-protected cushioned packaging."
            )

    # Deduplicate while preserving order
    unique_reasons = list(dict.fromkeys(reasons))
    return groups, unique_reasons
