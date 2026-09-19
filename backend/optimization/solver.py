"""
PackingSolver — OR-Tools CP-SAT 3D bin-packing solver
======================================================

Selects the cheapest container from the standard catalog that can
physically accommodate *all* supplied items, respecting:

- 3D spatial placement with 6 axis-aligned rotations per item
  (2 rotations for orientation-sensitive items)
- Pairwise no-overlap constraints between every item pair
- Per-container weight limits

The solver is invoked per candidate box (cheapest first); if CP-SAT
proves feasibility within the time limit the box is accepted.  If no
standard box is feasible the existing custom-crate / multi-truck
fallback fires.
"""

from ortools.sat.python import cp_model

# Per-box CP-SAT wall-clock timeout in seconds.
_SOLVER_TIMEOUT_S = 5.0
_DIMENSION_SCALE = 10


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

    # ------------------------------------------------------------------
    # Quick pre-filter (unchanged logic, used as a fast gate)
    # ------------------------------------------------------------------
    @staticmethod
    def _item_fits_box(item, bl, bw, bh):
        """Check whether a single item's dimensions fit inside the box.

        For orientation-sensitive items the height axis is locked; only
        length and width may be swapped.  For non-sensitive items all
        three axes may be freely rotated.
        """
        if item.orientation_sensitive:
            if item.height_cm > bh:
                return False
            dims = sorted([item.length_cm, item.width_cm])
            box_dims = sorted([bl, bw])
            return dims[0] <= box_dims[0] and dims[1] <= box_dims[1]
        else:
            dims = sorted([item.length_cm, item.width_cm, item.height_cm])
            box_dims = sorted([bl, bw, bh])
            return dims[0] <= box_dims[0] and dims[1] <= box_dims[1] and dims[2] <= box_dims[2]

    # ------------------------------------------------------------------
    # Six axis-aligned rotations of (L, W, H)
    # ------------------------------------------------------------------
    @staticmethod
    def _rotations(l, w, h):
        """Return the 6 unique axis-aligned rotations as (dx, dy, dz)."""
        return [
            (l, w, h),  # 0
            (l, h, w),  # 1
            (w, l, h),  # 2
            (w, h, l),  # 3
            (h, l, w),  # 4
            (h, w, l),  # 5
        ]

    @staticmethod
    def _allowed_rotations(item):
        """Return list of rotation indices allowed for *item*.

        Orientation-sensitive items keep their original height on the
        z-axis, so only rotations 0 (L,W,H) and 2 (W,L,H) are valid.
        """
        if item.orientation_sensitive:
            return [0, 2]
        return list(range(6))

    # ------------------------------------------------------------------
    # CP-SAT feasibility check for a single candidate box
    # ------------------------------------------------------------------
    def _cpsat_fits(self, items, bl, bw, bh):
        """Return True if CP-SAT proves all *items* fit inside (bl, bw, bh).

        Builds a 3D placement model with pairwise no-overlap disjunctive
        constraints and per-item rotation selection.
        """
        n = len(items)
        model = cp_model.CpModel()

        # --- Per-item decision variables -----------------------------------
        #   rot[i]       : which rotation is active
        #   x[i], y[i], z[i] : item corner position
        #   dx[i], dy[i], dz[i] : effective dims after rotation
        rot = []
        x, y, z = [], [], []
        dx, dy, dz = [], [], []

        for i, item in enumerate(items):
            il = round(item.length_cm * _DIMENSION_SCALE)
            iw = round(item.width_cm * _DIMENSION_SCALE)
            ih = round(item.height_cm * _DIMENSION_SCALE)

            all_rots = self._rotations(il, iw, ih)
            allowed = self._allowed_rotations(item)

            # Rotation selector
            rot_i = model.NewIntVar(0, 5, f"rot_{i}")
            # Restrict domain to allowed rotations
            model.AddAllowedAssignments([rot_i], [(r,) for r in allowed])
            rot.append(rot_i)

            # Effective dimensions — linked to rotation via element constraint
            dx_vals = [all_rots[r][0] for r in range(6)]
            dy_vals = [all_rots[r][1] for r in range(6)]
            dz_vals = [all_rots[r][2] for r in range(6)]

            max_dim = max(bl, bw, bh)
            dx_i = model.NewIntVar(1, max_dim, f"dx_{i}")
            dy_i = model.NewIntVar(1, max_dim, f"dy_{i}")
            dz_i = model.NewIntVar(1, max_dim, f"dz_{i}")

            model.AddElement(rot_i, dx_vals, dx_i)
            model.AddElement(rot_i, dy_vals, dy_i)
            model.AddElement(rot_i, dz_vals, dz_i)

            dx.append(dx_i)
            dy.append(dy_i)
            dz.append(dz_i)

            # Position (corner closest to origin)
            x_i = model.NewIntVar(0, bl, f"x_{i}")
            y_i = model.NewIntVar(0, bw, f"y_{i}")
            z_i = model.NewIntVar(0, bh, f"z_{i}")
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)

            # Containment: item must stay within box
            model.Add(x_i + dx_i <= bl)
            model.Add(y_i + dy_i <= bw)
            model.Add(z_i + dz_i <= bh)

        # --- Pairwise no-overlap -------------------------------------------
        for i in range(n):
            for j in range(i + 1, n):
                # At least one separating axis must hold
                b = [model.NewBoolVar(f"sep_{i}_{j}_{k}") for k in range(6)]

                model.Add(x[i] + dx[i] <= x[j]).OnlyEnforceIf(b[0])
                model.Add(x[j] + dx[j] <= x[i]).OnlyEnforceIf(b[1])
                model.Add(y[i] + dy[i] <= y[j]).OnlyEnforceIf(b[2])
                model.Add(y[j] + dy[j] <= y[i]).OnlyEnforceIf(b[3])
                model.Add(z[i] + dz[i] <= z[j]).OnlyEnforceIf(b[4])
                model.Add(z[j] + dz[j] <= z[i]).OnlyEnforceIf(b[5])

                model.AddBoolOr(b)

        # --- Solve ---------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = _SOLVER_TIMEOUT_S
        status = solver.Solve(model)

        return status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

    # ------------------------------------------------------------------
    # Public API (return shape unchanged)
    # ------------------------------------------------------------------
    def solve_3d_bin_packing(self, items: list, cost_matrix: dict):
        """Select the cheapest container(s) that can hold all *items*.

        Uses OR-Tools CP-SAT to verify 3D placement feasibility for each
        candidate container, with 6 axis-aligned rotations per item and
        pairwise no-overlap constraints.

        Returns a list of dicts sorted by box volume (ascending), each
        with keys: ``box_id``, ``b_cost``, ``box_vol``, ``utilization_pct``.
        """
        if not items:
            return [{"box_id": "NO-BOX", "b_cost": 0.0, "box_vol": 0, "utilization_pct": 0.0}]

        total_vol = sum(
            i.length_cm * i.width_cm * i.height_cm for i in items
        )
        total_weight = sum(i.weight_kg for i in items)

        valid_boxes = []
        for box_id, bl, bw, bh, b_wt, b_cost in self.boxes:
            box_vol = bl * bw * bh

            # --- Fast pre-filter (skip obviously infeasible) ---------------
            if total_weight > b_wt:
                continue
            if not all(self._item_fits_box(item, bl, bw, bh) for item in items):
                continue
            if total_vol > box_vol:
                continue

            # --- CP-SAT feasibility check ----------------------------------
            if len(items) == 1:
                # Single item trivially fits if pre-filter passed
                feasible = True
            else:
                feasible = self._cpsat_fits(
                    items,
                    round(bl * _DIMENSION_SCALE),
                    round(bw * _DIMENSION_SCALE),
                    round(bh * _DIMENSION_SCALE),
                )

            if feasible:
                utilization = min(0.95, total_vol / box_vol)
                valid_boxes.append({
                    "box_id": box_id,
                    "b_cost": b_cost,
                    "box_vol": box_vol,
                    "utilization_pct": utilization,
                })

        if valid_boxes:
            return sorted(valid_boxes, key=lambda x: x["box_vol"])

        # --- Fallback: no standard container fits --------------------------
        required_vol = total_vol / 0.75  # conservative estimate for custom
        custom_vol = required_vol * 1.1  # 10 % safety margin

        if custom_vol > 76_000_000:
            # Multi-truck scenario (each 40-ft HC ~= 76 m**3)
            trucks_needed = int(custom_vol // 76_000_000) + 1
            custom_cost = trucks_needed * 20_000.0
            utilization = min(0.95, total_vol / (trucks_needed * 76_000_000))
            return [{
                "box_id": f"MULTI-TRUCK({trucks_needed}x40ft)",
                "b_cost": custom_cost,
                "box_vol": trucks_needed * 76_000_000,
                "utilization_pct": utilization,
            }]

        # Custom crate — cap at Rs.5,000 to avoid runaway pricing
        custom_cost = min((custom_vol / 1000.0) * 22.5, 5000.0)
        utilization = min(0.95, total_vol / custom_vol)
        return [{
            "box_id": "CUSTOM-CRATE",
            "b_cost": custom_cost,
            "box_vol": custom_vol,
            "utilization_pct": utilization,
        }]
