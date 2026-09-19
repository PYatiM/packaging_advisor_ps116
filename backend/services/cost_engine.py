class DynamicCostEngine:
    def __init__(self):
        pass

    def get_current_rates(self):
        """
        Illustrative assumed domestic parcel rates for this prototype, as of
        2025-01-15: Rs. 60/kg for air and Rs. 15/kg for surface. These values
        are calibrated to be broadly consistent with private courier per-kg
        pricing for small parcels in India, but are not tied to any single
        published rate card. They are configurable via
        backend.core.config.Settings and can be overridden with environment
        variables.

        Cost model:
          - Cartons (C-*): "material" is the box cost; shipping uses
            per-kg courier rates from the router.
          - Pallets / PTL: "material" is pallet + handling cost;
            "flat_rate" is the per-consignment lane rate (applied
            before distance multiplier).
          - Trucks (FTL): "material" is loading, lashing, and crating;
            "flat_rate" is the base freight rate for the lane.
        """
        return {
            # --- Courier cartons (per-kg shipping via router) ---
            "C-SMALL-01":     {"material":   40, "shipping_flat": False},
            "C-MED-05":       {"material":   95, "shipping_flat": False},
            "C-LARGE-99":     {"material":  210, "shipping_flat": False},
            "C-XL-100":       {"material":  400, "shipping_flat": False},

            # --- PTL (Part Truck Load) — flat lane rates ---
            "PALLET-STD":     {"material":  500, "shipping_flat": True, "flat_rate": 3500},
            "PTL-REEFER":     {"material": 1500, "shipping_flat": True, "flat_rate": 4500},

            # --- FTL (Full Truck Load) — flat lane rates ---
            "TRUCK-20FT":     {"material": 5000, "shipping_flat": True, "flat_rate": 18000},
            "TRUCK-40FT-HC":  {"material": 8000, "shipping_flat": True, "flat_rate": 28000},
        }
