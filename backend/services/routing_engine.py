"""Indian PIN-code and carrier shipping-zone routing engine.

Models Delhivery / Blue Dart / Shiprocket style shipping zones (Zone A–E)
from 6-digit Indian postal PIN codes, providing distance multipliers,
handling-node counts, transit time ranges, and ODA / special-zone flags.
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple


@dataclass(frozen=True)
class RouteProfile:
    """Derived carrier route characteristics between origin and destination PINs."""

    source_pin: str
    destination_pin: str
    source_state: str
    destination_state: str
    source_zone: str
    destination_zone: str
    zone_code: str
    shipping_zone_tier: str
    distance_multiplier: float
    handling_nodes: int
    surface_days_min: int
    surface_days_max: int
    air_days_min: int
    air_days_max: int
    is_same_subzone: bool
    is_oda_special: bool
    estimated_distance_km: float = 1000.0


@dataclass(frozen=True)
class _PinMetadata:
    state: str
    macro_zone: str
    metro_cluster: str
    is_special_oda: bool


_METRO_CLUSTERS: Dict[str, str] = {
    "110": "Delhi NCR",
    "121": "Delhi NCR",
    "122": "Delhi NCR",
    "201": "Delhi NCR",
    "400": "Mumbai MMR",
    "401": "Mumbai MMR",
    "410": "Mumbai MMR",
    "411": "Pune",
    "380": "Ahmedabad",
    "560": "Bengaluru",
    "500": "Hyderabad",
    "600": "Chennai",
    "700": "Kolkata",
}

_TWO_DIGIT_PREFIX_MAP: Dict[str, Tuple[str, str, bool]] = {
    "11": ("Delhi", "North", False),
    "12": ("Haryana", "North", False),
    "13": ("Haryana", "North", False),
    "14": ("Punjab", "North", False),
    "15": ("Punjab", "North", False),
    "16": ("Chandigarh", "North", False),
    "17": ("Himachal Pradesh", "North", True),
    "18": ("Jammu & Kashmir", "North", True),
    "19": ("Jammu & Kashmir / Ladakh", "North", True),
    "20": ("Uttar Pradesh", "North", False),
    "21": ("Uttar Pradesh", "North", False),
    "22": ("Uttar Pradesh", "North", False),
    "23": ("Uttar Pradesh", "North", False),
    "24": ("Uttarakhand / UP", "North", False),
    "25": ("Uttar Pradesh", "North", False),
    "26": ("Uttarakhand / UP", "North", False),
    "27": ("Uttar Pradesh", "North", False),
    "28": ("Uttar Pradesh", "North", False),
    "30": ("Rajasthan", "West", False),
    "31": ("Rajasthan", "West", False),
    "32": ("Rajasthan", "West", False),
    "33": ("Rajasthan", "West", False),
    "34": ("Rajasthan", "West", False),
    "36": ("Gujarat", "West", False),
    "37": ("Gujarat", "West", False),
    "38": ("Gujarat", "West", False),
    "39": ("Gujarat", "West", False),
    "40": ("Maharashtra", "West", False),
    "41": ("Maharashtra", "West", False),
    "42": ("Maharashtra", "West", False),
    "43": ("Maharashtra", "West", False),
    "44": ("Maharashtra", "West", False),
    "45": ("Madhya Pradesh", "Central", False),
    "46": ("Madhya Pradesh", "Central", False),
    "47": ("Madhya Pradesh", "Central", False),
    "48": ("Madhya Pradesh", "Central", False),
    "49": ("Chhattisgarh", "Central", False),
    "50": ("Telangana", "South", False),
    "51": ("Andhra Pradesh", "South", False),
    "52": ("Andhra Pradesh", "South", False),
    "53": ("Andhra Pradesh", "South", False),
    "56": ("Karnataka", "South", False),
    "57": ("Karnataka", "South", False),
    "58": ("Karnataka", "South", False),
    "59": ("Karnataka", "South", False),
    "60": ("Tamil Nadu", "South", False),
    "61": ("Tamil Nadu", "South", False),
    "62": ("Tamil Nadu", "South", False),
    "63": ("Tamil Nadu", "South", False),
    "64": ("Tamil Nadu", "South", False),
    "65": ("Tamil Nadu", "South", False),
    "66": ("Tamil Nadu", "South", False),
    "67": ("Kerala", "South", False),
    "68": ("Kerala", "South", False),
    "69": ("Kerala", "South", False),
    "70": ("West Bengal", "East", False),
    "71": ("West Bengal", "East", False),
    "72": ("West Bengal", "East", False),
    "73": ("West Bengal", "East", False),
    "74": ("West Bengal", "East", False),
    "75": ("Odisha", "East", False),
    "76": ("Odisha", "East", False),
    "77": ("Odisha", "East", False),
    "78": ("Assam", "North-East", True),
    "79": ("North-East States", "North-East", True),
    "80": ("Bihar", "East", False),
    "81": ("Bihar / Jharkhand", "East", False),
    "82": ("Jharkhand / Bihar", "East", False),
    "83": ("Jharkhand", "East", False),
    "84": ("Bihar", "East", False),
    "85": ("Bihar", "East", False),
}

_ADJACENT_STATES: Dict[str, Set[str]] = {
    "Delhi": {"Haryana", "Uttar Pradesh", "Uttarakhand / UP"},
    "Haryana": {"Delhi", "Punjab", "Chandigarh", "Rajasthan", "Uttar Pradesh", "Uttarakhand / UP", "Himachal Pradesh"},
    "Punjab": {"Haryana", "Chandigarh", "Himachal Pradesh", "Rajasthan", "Jammu & Kashmir"},
    "Chandigarh": {"Punjab", "Haryana", "Himachal Pradesh"},
    "Uttar Pradesh": {"Delhi", "Haryana", "Rajasthan", "Madhya Pradesh", "Chhattisgarh", "Bihar", "Jharkhand", "Bihar / Jharkhand", "Uttarakhand / UP"},
    "Uttarakhand / UP": {"Delhi", "Haryana", "Uttar Pradesh", "Himachal Pradesh"},
    "Rajasthan": {"Haryana", "Punjab", "Uttar Pradesh", "Madhya Pradesh", "Gujarat"},
    "Gujarat": {"Rajasthan", "Madhya Pradesh", "Maharashtra"},
    "Maharashtra": {"Gujarat", "Madhya Pradesh", "Chhattisgarh", "Telangana", "Karnataka"},
    "Madhya Pradesh": {"Rajasthan", "Uttar Pradesh", "Chhattisgarh", "Maharashtra", "Gujarat"},
    "Chhattisgarh": {"Madhya Pradesh", "Uttar Pradesh", "Jharkhand", "Odisha", "Telangana", "Maharashtra"},
    "Telangana": {"Maharashtra", "Chhattisgarh", "Odisha", "Andhra Pradesh", "Karnataka"},
    "Andhra Pradesh": {"Telangana", "Odisha", "Chhattisgarh", "Karnataka", "Tamil Nadu"},
    "Karnataka": {"Maharashtra", "Telangana", "Andhra Pradesh", "Tamil Nadu", "Kerala"},
    "Tamil Nadu": {"Andhra Pradesh", "Karnataka", "Kerala"},
    "Kerala": {"Karnataka", "Tamil Nadu"},
    "West Bengal": {"Odisha", "Jharkhand", "Bihar", "Bihar / Jharkhand", "Jharkhand / Bihar", "Assam"},
    "Odisha": {"West Bengal", "Jharkhand", "Jharkhand / Bihar", "Chhattisgarh", "Andhra Pradesh"},
    "Bihar": {"Uttar Pradesh", "West Bengal", "Jharkhand", "Bihar / Jharkhand", "Jharkhand / Bihar"},
    "Jharkhand": {"Bihar", "Uttar Pradesh", "Chhattisgarh", "Odisha", "West Bengal", "Bihar / Jharkhand", "Jharkhand / Bihar"},
    "Bihar / Jharkhand": {"Uttar Pradesh", "Bihar", "Jharkhand", "West Bengal", "Odisha"},
    "Jharkhand / Bihar": {"Uttar Pradesh", "Bihar", "Jharkhand", "West Bengal", "Odisha"},
}

_OPPOSITE_MACRO_ZONE_PAIRS: Set[frozenset] = {
    frozenset({"North", "South"}),
    frozenset({"East", "West"}),
}


class RoutingEngine:
    """Resolve Indian 6-digit PIN routes into carrier shipping zones."""

    @staticmethod
    def sanitize_pin(pin: object, default_pin: str) -> str:
        text = str(pin).strip() if pin is not None else ""
        if len(text) == 6 and text.isdigit():
            return text
        if len(text) >= 2 and text.isdigit():
            return text.ljust(6, "0")[:6]
        return default_pin

    @staticmethod
    def resolve_pin_metadata(pin: str) -> _PinMetadata:
        prefix5 = pin[:5]
        prefix3 = pin[:3]
        prefix2 = pin[:2]
        first = pin[0]

        metro = _METRO_CLUSTERS.get(prefix3, "")

        if prefix5 == "68255":
            return _PinMetadata("Lakshadweep", "South", metro, True)
        if prefix3 == "744":
            return _PinMetadata("Andaman & Nicobar", "East", metro, True)
        if prefix3 == "403":
            return _PinMetadata("Goa", "West", metro, False)
        if first == "9":
            return _PinMetadata("APS / Army Postal", "APS", metro, True)

        if prefix2 in _TWO_DIGIT_PREFIX_MAP:
            state, macro_zone, is_special = _TWO_DIGIT_PREFIX_MAP[prefix2]
            return _PinMetadata(state, macro_zone, metro, is_special)

        fallback_by_first = {
            "1": ("North Region", "North", False),
            "2": ("North Region", "North", False),
            "3": ("West Region", "West", False),
            "4": ("West / Central", "West", False),
            "5": ("South Region", "South", False),
            "6": ("South Region", "South", False),
            "7": ("East Region", "East", False),
            "8": ("East Region", "East", False),
            "9": ("APS / Army Postal", "APS", True),
        }
        state, macro_zone, is_special = fallback_by_first.get(
            first, ("Unknown", "Unknown", False)
        )
        return _PinMetadata(state, macro_zone, metro, is_special)

    def analyze_route(self, source_pin: str, destination_pin: str) -> RouteProfile:
        """Classify origin -> destination lane into Zone A–E with SLA & cost factors."""
        src = self.sanitize_pin(source_pin, "110001")
        dst = self.sanitize_pin(destination_pin, "560001")

        src_meta = self.resolve_pin_metadata(src)
        dst_meta = self.resolve_pin_metadata(dst)

        # Zone E: Either endpoint is a special / hill / island / North-East / APS region
        if src_meta.is_special_oda or dst_meta.is_special_oda:
            return RouteProfile(
                source_pin=src,
                destination_pin=dst,
                source_state=src_meta.state,
                destination_state=dst_meta.state,
                source_zone=src_meta.macro_zone,
                destination_zone=dst_meta.macro_zone,
                zone_code="E",
                shipping_zone_tier="Zone E (Special / North-East & J&K)",
                distance_multiplier=2.85,
                handling_nodes=7,
                surface_days_min=6,
                surface_days_max=10,
                air_days_min=2,
                air_days_max=3,
                is_same_subzone=False,
                is_oda_special=True,
                estimated_distance_km=2450.0,
            )

        # Zone A: Intra-city (same 3-digit prefix or same metro cluster)
        same_metro = (
            bool(src_meta.metro_cluster)
            and src_meta.metro_cluster == dst_meta.metro_cluster
        )
        if src[:3] == dst[:3] or same_metro:
            return RouteProfile(
                source_pin=src,
                destination_pin=dst,
                source_state=src_meta.state,
                destination_state=dst_meta.state,
                source_zone=src_meta.macro_zone,
                destination_zone=dst_meta.macro_zone,
                zone_code="A",
                shipping_zone_tier="Zone A (Local / Metro Cluster)",
                distance_multiplier=1.0,
                handling_nodes=2,
                surface_days_min=1,
                surface_days_max=2,
                air_days_min=1,
                air_days_max=1,
                is_same_subzone=True,
                is_oda_special=False,
                estimated_distance_km=35.0,
            )

        # Zone B (strictly intra-state / same 2-digit prefix)
        same_state = src_meta.state == dst_meta.state and src_meta.state != "Unknown"
        if same_state or src[:2] == dst[:2]:
            return RouteProfile(
                source_pin=src,
                destination_pin=dst,
                source_state=src_meta.state,
                destination_state=dst_meta.state,
                source_zone=src_meta.macro_zone,
                destination_zone=dst_meta.macro_zone,
                zone_code="B",
                shipping_zone_tier="Zone B (Regional / Intra-State)",
                distance_multiplier=1.30,
                handling_nodes=3,
                surface_days_min=2,
                surface_days_max=3,
                air_days_min=1,
                air_days_max=2,
                is_same_subzone=False,
                is_oda_special=False,
                estimated_distance_km=320.0,
            )

        # Zone C: Metro-to-Metro corridor across different states
        both_metro = bool(src_meta.metro_cluster) and bool(dst_meta.metro_cluster)
        if both_metro:
            return RouteProfile(
                source_pin=src,
                destination_pin=dst,
                source_state=src_meta.state,
                destination_state=dst_meta.state,
                source_zone=src_meta.macro_zone,
                destination_zone=dst_meta.macro_zone,
                zone_code="C",
                shipping_zone_tier="Zone C (Metro-to-Metro Corridor)",
                distance_multiplier=1.65,
                handling_nodes=4,
                surface_days_min=3,
                surface_days_max=4,
                air_days_min=1,
                air_days_max=2,
                is_same_subzone=False,
                is_oda_special=False,
                estimated_distance_km=1450.0,
            )

        # Zone B (adjacent states when not a metro-to-metro corridor)
        is_adjacent_state = dst_meta.state in _ADJACENT_STATES.get(src_meta.state, set())
        if is_adjacent_state:
            return RouteProfile(
                source_pin=src,
                destination_pin=dst,
                source_state=src_meta.state,
                destination_state=dst_meta.state,
                source_zone=src_meta.macro_zone,
                destination_zone=dst_meta.macro_zone,
                zone_code="B",
                shipping_zone_tier="Zone B (Regional / Neighboring State)",
                distance_multiplier=1.30,
                handling_nodes=3,
                surface_days_min=2,
                surface_days_max=3,
                air_days_min=1,
                air_days_max=2,
                is_same_subzone=False,
                is_oda_special=False,
                estimated_distance_km=320.0,
            )

        # Zone D: National cross-country
        pair = frozenset({src_meta.macro_zone, dst_meta.macro_zone})
        is_opposite = pair in _OPPOSITE_MACRO_ZONE_PAIRS
        dist_mult = 2.35 if is_opposite else 2.05
        return RouteProfile(
            source_pin=src,
            destination_pin=dst,
            source_state=src_meta.state,
            destination_state=dst_meta.state,
            source_zone=src_meta.macro_zone,
            destination_zone=dst_meta.macro_zone,
            zone_code="D",
            shipping_zone_tier="Zone D (National Cross-Country)",
            distance_multiplier=dist_mult,
            handling_nodes=5,
            surface_days_min=4,
            surface_days_max=6,
            air_days_min=2,
            air_days_max=2,
            is_same_subzone=False,
            is_oda_special=False,
            estimated_distance_km=2150.0 if is_opposite else 1750.0,
        )
