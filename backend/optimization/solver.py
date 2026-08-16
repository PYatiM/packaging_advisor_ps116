class PackingSolver:
    def __init__(self):
        # Available containers sorted small-to-large:
        #   (ID, inner_L_cm, inner_W_cm, inner_H_cm, max_weight_kg, material_cost_INR)
        self.boxes = [
            ("C-SMALL-01",    15,   15,   15,      2.0,    100.00),
            ("C-MED-05",      30,   30,   20,      5.0,    225.00),
            ("C-LARGE-99",    50,   50,   40,     15.0,    460.00),
            ("C-XL-100",     100,  100,  100,     50.0,    850.00),
            ("PALLET-STD",   120,  100,  150,   1000.0,   1500.00),
            ("PTL-REEFER",   590,  235,  239,   5000.0,   5000.00),
            ("TRUCK-20FT",   590,  235,  239,  15000.0,  10000.00),
            ("TRUCK-40FT-HC",1203, 235,  270,  28000.0,  15000.00),
        ]

    def _item_fits_box(self, item, bl, bw, bh):
        """Check whether a single item's dimensions fit inside the box.

        For orientation-sensitive items the height axis is locked; only
        length and width may be swapped.  For non-sensitive items all
        three axes may be freely rotated.
        """
        if item.orientation_sensitive:
            # Height is fixed; try both L/W orientations
            if item.height_cm > bh:
                return False
            dims = sorted([item.length_cm, item.width_cm])
            box_dims = sorted([bl, bw])
            return dims[0] <= box_dims[0] and dims[1] <= box_dims[1]
        else:
            # Free rotation — sort both and compare rank-by-rank
            dims = sorted([item.length_cm, item.width_cm, item.height_cm])
            box_dims = sorted([bl, bw, bh])
            return dims[0] <= box_dims[0] and dims[1] <= box_dims[1] and dims[2] <= box_dims[2]

    def solve_3d_bin_packing(self, items: list, cost_matrix: dict):
        """Find the smallest container that can hold all *items*.

        Selection criteria (all must be satisfied):
          1. Every individual item must physically fit inside the box.
          2. Total weight must not exceed the box's weight limit.
          3. Total volume (with a 75 % packing-efficiency factor) must
             fit inside the box.
        """
        if not items:
            return [{"box_id": "NO-BOX", "b_cost": 0.0, "box_vol": 0, "utilization_pct": 0.0}]

        total_vol = sum(i.length_cm * i.width_cm * i.height_cm for i in items)
        total_weight = sum(i.weight_kg for i in items)

        # Assume ~75 % packing efficiency for irregular mixed shapes
        required_vol = total_vol / 0.75

        valid_boxes = []
        for box_id, bl, bw, bh, b_wt, b_cost in self.boxes:
            box_vol = bl * bw * bh

            # Volume and weight gates
            if required_vol > box_vol or total_weight > b_wt:
                continue

            # Every item must individually fit inside the box
            if not all(self._item_fits_box(item, bl, bw, bh) for item in items):
                continue

            utilization = min(0.95, total_vol / box_vol)
            valid_boxes.append({
                "box_id": box_id,
                "b_cost": b_cost,
                "box_vol": box_vol,
                "utilization_pct": utilization,
            })

        if valid_boxes:
            return sorted(valid_boxes, key=lambda x: x["box_vol"])

        # --- Fallback: no standard container fits ---
        custom_vol = required_vol * 1.1  # 10 % safety margin

        if custom_vol > 76_000_000:
            # Multi-truck scenario (each 40-ft HC ≈ 76 m³)
            trucks_needed = int(custom_vol // 76_000_000) + 1
            custom_cost = trucks_needed * 20_000.0
            utilization = min(0.95, total_vol / (trucks_needed * 76_000_000))
            return [{
                "box_id": f"MULTI-TRUCK({trucks_needed}x40ft)",
                "b_cost": custom_cost,
                "box_vol": trucks_needed * 76_000_000,
                "utilization_pct": utilization,
            }]

        # Custom crate — cap at ₹5,000 to avoid runaway pricing
        custom_cost = min((custom_vol / 1000.0) * 22.5, 5000.0)
        utilization = min(0.95, total_vol / custom_vol)
        return [{
            "box_id": "CUSTOM-CRATE",
            "b_cost": custom_cost,
            "box_vol": custom_vol,
            "utilization_pct": utilization,
        }]
