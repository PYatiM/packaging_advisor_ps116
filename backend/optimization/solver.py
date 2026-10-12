"""PackingSolver — OR-Tools CP-SAT 3D bin-packing solver with ISTA cushioning.

Selects feasible single containers or multi-carton parcel splits from the
IS 2771 / cold-chain catalog, respecting:
- Unit quantity expansion (SKUItem.quantity)
- ISTA 3A per-face protective cushion clearance around fragile items
- 3D spatial placement with 6 axis-aligned rotations (2 upright rotations
  for orientation-sensitive items)
- Pairwise 3D non-overlap constraints + gravity-weighted corner packing
- Per-container payload limits
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ortools.sat.python import cp_model

from backend.core.catalog import (
    CONTAINER_CATALOG,
    ORDERED_CONTAINERS,
    ContainerSpec,
    compute_cushion_pad_cm,
)

_SOLVER_TIMEOUT_S = 0.25
_DIMENSION_SCALE = 10
_MAX_CPSAT_UNITS = 35
_MAX_SPLIT_BINS = 5


@dataclass
class _PackUnit:
    sku_id: str
    product_category: str
    raw_l: float
    raw_w: float
    raw_h: float
    pad_cm: float
    l: float
    w: float
    h: float
    weight_kg: float
    fragility_score: float
    orientation_sensitive: bool
    group_key: str


class PackingSolver:
    """3D constraint-programming container packing and multi-carton splitter."""

    def __init__(self):
        self.catalog = CONTAINER_CATALOG
        self.ordered_specs = ORDERED_CONTAINERS
        # Legacy tuple list preserved for external scripts
        self.boxes = [
            (
                s.container_id,
                s.inner_dims_cm[0],
                s.inner_dims_cm[1],
                s.inner_dims_cm[2],
                s.max_payload_kg,
                s.material_cost_inr,
            )
            for s in ORDERED_CONTAINERS
        ]

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
    def _allowed_rotations(item) -> List[int]:
        """Keep original height on z-axis for orientation-sensitive items."""
        if getattr(item, "orientation_sensitive", False):
            return [0, 2]
        return list(range(6))

    @staticmethod
    def _unit_fits_dims(
        l: float, w: float, h: float, orientation_sensitive: bool, bl: float, bw: float, bh: float
    ) -> bool:
        if orientation_sensitive:
            if h > bh + 1e-6:
                return False
            lw = sorted([l, w])
            blw = sorted([bl, bw])
            return lw[0] <= blw[0] + 1e-6 and lw[1] <= blw[1] + 1e-6
        dims = sorted([l, w, h])
        box_dims = sorted([bl, bw, bh])
        return (
            dims[0] <= box_dims[0] + 1e-6
            and dims[1] <= box_dims[1] + 1e-6
            and dims[2] <= box_dims[2] + 1e-6
        )

    @staticmethod
    def _item_fits_box(item, bl: float, bw: float, bh: float) -> bool:
        """Check whether a single item's raw dimensions fit inside (bl, bw, bh) in cm."""
        return PackingSolver._unit_fits_dims(
            item.length_cm,
            item.width_cm,
            item.height_cm,
            getattr(item, "orientation_sensitive", False),
            bl,
            bw,
            bh,
        )

    @staticmethod
    def _expand_units(items: list, use_padding: bool = True) -> List[_PackUnit]:
        units: List[_PackUnit] = []
        for item in items:
            qty = max(1, int(getattr(item, "quantity", 1)))
            pad = (
                compute_cushion_pad_cm(item.fragility_score, item.weight_kg)
                if use_padding
                else 0.0
            )
            pl = item.length_cm + 2.0 * pad
            pw = item.width_cm + 2.0 * pad
            ph = item.height_cm + 2.0 * pad
            gkey = (
                f"{item.sku_id}|{pl:.2f}|{pw:.2f}|{ph:.2f}|"
                f"{item.weight_kg:.3f}|{int(item.orientation_sensitive)}"
            )
            for q in range(qty):
                uid = item.sku_id if qty == 1 else f"{item.sku_id}#{q + 1}"
                units.append(
                    _PackUnit(
                        sku_id=uid,
                        product_category=getattr(item, "product_category", "General"),
                        raw_l=item.length_cm,
                        raw_w=item.width_cm,
                        raw_h=item.height_cm,
                        pad_cm=pad,
                        l=pl,
                        w=pw,
                        h=ph,
                        weight_kg=item.weight_kg,
                        fragility_score=item.fragility_score,
                        orientation_sensitive=getattr(item, "orientation_sensitive", False),
                        group_key=gkey,
                    )
                )
        return units

    def _shelf_ffd_pack(
        self,
        units: List[_PackUnit],
        bl_cm: float,
        bw_cm: float,
        bh_cm: float,
        bin_index: int = 0,
    ) -> Optional[List[dict]]:
        """Fast deterministic 3D shelf First-Fit Decreasing packer for bulk carts."""
        sorted_units = sorted(units, key=lambda u: (u.h, u.l * u.w), reverse=True)
        placements: List[dict] = []

        z_cursor = 0.0
        y_cursor = 0.0
        x_cursor = 0.0
        row_depth = 0.0
        layer_height = 0.0

        for u in sorted_units:
            rots = self._rotations(u.l, u.w, u.h)
            allowed = self._allowed_rotations(u)
            candidates = [
                rots[r]
                for r in allowed
                if rots[r][0] <= bl_cm + 1e-6
                and rots[r][1] <= bw_cm + 1e-6
                and rots[r][2] <= bh_cm + 1e-6
            ]
            if not candidates:
                return None

            # Prefer lowest height for stable stacking
            candidates.sort(key=lambda d: (d[2], d[1], d[0]))
            placed = False

            for dx, dy, dz in candidates:
                # 1. Current row
                if (
                    x_cursor + dx <= bl_cm + 1e-6
                    and y_cursor + dy <= bw_cm + 1e-6
                    and z_cursor + dz <= bh_cm + 1e-6
                ):
                    placements.append(
                        self._make_placement_dict(
                            u, x_cursor, y_cursor, z_cursor, dx, dy, dz, bin_index
                        )
                    )
                    x_cursor += dx
                    row_depth = max(row_depth, dy)
                    layer_height = max(layer_height, dz)
                    placed = True
                    break

                # 2. Next row on same z-layer
                next_y = y_cursor + row_depth
                if (
                    dx <= bl_cm + 1e-6
                    and next_y + dy <= bw_cm + 1e-6
                    and z_cursor + dz <= bh_cm + 1e-6
                ):
                    x_cursor = dx
                    y_cursor = next_y
                    row_depth = dy
                    layer_height = max(layer_height, dz)
                    placements.append(
                        self._make_placement_dict(
                            u, 0.0, y_cursor, z_cursor, dx, dy, dz, bin_index
                        )
                    )
                    placed = True
                    break

                # 3. Next z-layer
                next_z = z_cursor + layer_height
                if (
                    dx <= bl_cm + 1e-6
                    and dy <= bw_cm + 1e-6
                    and next_z + dz <= bh_cm + 1e-6
                ):
                    z_cursor = next_z
                    x_cursor = dx
                    y_cursor = 0.0
                    row_depth = dy
                    layer_height = dz
                    placements.append(
                        self._make_placement_dict(
                            u, 0.0, 0.0, z_cursor, dx, dy, dz, bin_index
                        )
                    )
                    placed = True
                    break

            if not placed:
                return None

        return placements

    @staticmethod
    def _make_placement_dict(
        u: _PackUnit,
        x_cm: float,
        y_cm: float,
        z_cm: float,
        dx_cm: float,
        dy_cm: float,
        dz_cm: float,
        bin_index: int,
    ) -> dict:
        return {
            "sku_id": u.sku_id,
            "product_category": u.product_category,
            "x_cm": round(x_cm, 2),
            "y_cm": round(y_cm, 2),
            "z_cm": round(z_cm, 2),
            "dx_cm": round(dx_cm, 2),
            "dy_cm": round(dy_cm, 2),
            "dz_cm": round(dz_cm, 2),
            "pad_cm": round(u.pad_cm, 2),
            "weight_kg": round(u.weight_kg, 3),
            "fragility_score": round(u.fragility_score, 3),
            "bin_index": bin_index,
        }

    def _cpsat_pack(
        self,
        units: List[_PackUnit],
        bl_cm: float,
        bw_cm: float,
        bh_cm: float,
        bin_index: int = 0,
    ) -> Optional[List[dict]]:
        """Solve 3D placement with OR-Tools CP-SAT and return coordinates if feasible."""
        if not units:
            return []

        if not all(
            self._unit_fits_dims(
                u.l, u.w, u.h, u.orientation_sensitive, bl_cm, bw_cm, bh_cm
            )
            for u in units
        ):
            return None

        total_unit_vol = sum(u.l * u.w * u.h for u in units)
        if total_unit_vol > bl_cm * bw_cm * bh_cm + 1e-3:
            return None

        if len(units) == 1:
            u = units[0]
            rots = self._rotations(u.l, u.w, u.h)
            allowed = self._allowed_rotations(u)
            valid = [
                rots[r]
                for r in allowed
                if rots[r][0] <= bl_cm + 1e-6
                and rots[r][1] <= bw_cm + 1e-6
                and rots[r][2] <= bh_cm + 1e-6
            ]
            if not valid:
                return None
            # Prefer lowest center-of-gravity height
            dx, dy, dz = min(valid, key=lambda d: (d[2], d[1], d[0]))
            return [self._make_placement_dict(u, 0.0, 0.0, 0.0, dx, dy, dz, bin_index)]

        if len(units) > _MAX_CPSAT_UNITS:
            return self._shelf_ffd_pack(units, bl_cm, bw_cm, bh_cm, bin_index)

        BL = max(1, round(bl_cm * _DIMENSION_SCALE))
        BW = max(1, round(bw_cm * _DIMENSION_SCALE))
        BH = max(1, round(bh_cm * _DIMENSION_SCALE))

        n = len(units)
        model = cp_model.CpModel()

        x, y, z = [], [], []
        dx, dy, dz = [], [], []

        for i, u in enumerate(units):
            il = max(1, round(u.l * _DIMENSION_SCALE))
            iw = max(1, round(u.w * _DIMENSION_SCALE))
            ih = max(1, round(u.h * _DIMENSION_SCALE))

            all_rots = self._rotations(il, iw, ih)
            allowed = [
                r
                for r in self._allowed_rotations(u)
                if all_rots[r][0] <= BL
                and all_rots[r][1] <= BW
                and all_rots[r][2] <= BH
            ]
            if not allowed:
                return None

            rot_i = model.NewIntVar(0, 5, f"rot_{i}")
            model.AddAllowedAssignments([rot_i], [(r,) for r in allowed])

            dx_vals = [all_rots[r][0] for r in range(6)]
            dy_vals = [all_rots[r][1] for r in range(6)]
            dz_vals = [all_rots[r][2] for r in range(6)]

            max_dim = max(BL, BW, BH, il, iw, ih)
            dx_i = model.NewIntVar(1, max_dim, f"dx_{i}")
            dy_i = model.NewIntVar(1, max_dim, f"dy_{i}")
            dz_i = model.NewIntVar(1, max_dim, f"dz_{i}")

            model.AddElement(rot_i, dx_vals, dx_i)
            model.AddElement(rot_i, dy_vals, dy_i)
            model.AddElement(rot_i, dz_vals, dz_i)

            dx.append(dx_i)
            dy.append(dy_i)
            dz.append(dz_i)

            x_i = model.NewIntVar(0, BL, f"x_{i}")
            y_i = model.NewIntVar(0, BW, f"y_{i}")
            z_i = model.NewIntVar(0, BH, f"z_{i}")
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)

            model.Add(x_i + dx_i <= BL)
            model.Add(y_i + dy_i <= BW)
            model.Add(z_i + dz_i <= BH)

            if i > 0 and u.group_key == units[i - 1].group_key:
                model.Add(z[i - 1] <= z_i)

        # Pairwise non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                b = [model.NewBoolVar(f"sep_{i}_{j}_{k}") for k in range(6)]
                model.Add(x[i] + dx[i] <= x[j]).OnlyEnforceIf(b[0])
                model.Add(x[j] + dx[j] <= x[i]).OnlyEnforceIf(b[1])
                model.Add(y[i] + dy[i] <= y[j]).OnlyEnforceIf(b[2])
                model.Add(y[j] + dy[j] <= y[i]).OnlyEnforceIf(b[3])
                model.Add(z[i] + dz[i] <= z[j]).OnlyEnforceIf(b[4])
                model.Add(z[j] + dz[j] <= z[i]).OnlyEnforceIf(b[5])
                model.AddBoolOr(b)

        # Gravity + compact origin corner objective
        model.Minimize(
            sum(
                max(1, round(u.weight_kg * 10)) * (z[i] * 8 + x[i] + y[i])
                for i, u in enumerate(units)
            )
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = _SOLVER_TIMEOUT_S
        solver.parameters.relative_gap_limit = 0.10
        solver.parameters.num_search_workers = 4 if n > 6 else 1
        status = solver.Solve(model)

        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return [
                self._make_placement_dict(
                    units[i],
                    solver.Value(x[i]) / _DIMENSION_SCALE,
                    solver.Value(y[i]) / _DIMENSION_SCALE,
                    solver.Value(z[i]) / _DIMENSION_SCALE,
                    solver.Value(dx[i]) / _DIMENSION_SCALE,
                    solver.Value(dy[i]) / _DIMENSION_SCALE,
                    solver.Value(dz[i]) / _DIMENSION_SCALE,
                    bin_index,
                )
                for i in range(n)
            ]

        if status == cp_model.UNKNOWN:
            return self._shelf_ffd_pack(units, bl_cm, bw_cm, bh_cm, bin_index)

        return None

    def _cpsat_fits(self, items: list, bl: float, bw: float, bh: float) -> bool:
        """Backward-compatible boolean feasibility check in cm."""
        units = self._expand_units(items, use_padding=False)
        return self._cpsat_pack(units, bl, bw, bh) is not None

    def _try_multi_carton_split(
        self,
        units: List[_PackUnit],
        spec: ContainerSpec,
    ) -> Optional[List[dict]]:
        """Pack *units* into 2.._MAX_SPLIT_BINS identical bins of *spec*."""
        if len(units) < 2:
            return None

        bl, bw, bh = spec.inner_dims_cm
        box_vol = spec.inner_volume_cm3

        # Every individual unit must fit inside a single bin
        for u in units:
            if u.weight_kg > spec.max_payload_kg:
                return None
            if not self._unit_fits_dims(
                u.l, u.w, u.h, u.orientation_sensitive, bl, bw, bh
            ):
                return None

        sorted_units = sorted(
            units,
            key=lambda u: (u.l * u.w * u.h, u.weight_kg),
            reverse=True,
        )

        bins: List[List[_PackUnit]] = []
        bin_placements: List[List[dict]] = []

        for u in sorted_units:
            u_vol = u.l * u.w * u.h
            placed = False

            for b_idx, b_units in enumerate(bins):
                if sum(x.weight_kg for x in b_units) + u.weight_kg > spec.max_payload_kg:
                    continue
                if sum(x.l * x.w * x.h for x in b_units) + u_vol > box_vol + 1e-3:
                    continue
                trial = b_units + [u]
                packed = self._cpsat_pack(trial, bl, bw, bh, bin_index=b_idx)
                if packed is not None:
                    bins[b_idx] = trial
                    bin_placements[b_idx] = packed
                    placed = True
                    break

            if not placed:
                if len(bins) >= _MAX_SPLIT_BINS:
                    return None
                new_idx = len(bins)
                packed = self._cpsat_pack([u], bl, bw, bh, bin_index=new_idx)
                if packed is None:
                    return None
                bins.append([u])
                bin_placements.append(packed)

        if not (2 <= len(bins) <= _MAX_SPLIT_BINS):
            return None

        flattened: List[dict] = []
        for p_list in bin_placements:
            flattened.extend(p_list)
        return flattened

    def _compute_unpadded_cushion_ratio(
        self,
        units_padded: List[_PackUnit],
        spec: ContainerSpec,
    ) -> float:
        """Estimate cushion adequacy [0.15, 1.0] when forced to pack without full ISTA pads."""
        required_pads = [u.pad_cm for u in units_padded if u.pad_cm > 0]
        if not required_pads:
            return 1.0
        bl, bw, bh = spec.inner_dims_cm
        total_raw_vol = sum(u.raw_l * u.raw_w * u.raw_h for u in units_padded)
        free_ratio = max(0.0, 1.0 - (total_raw_vol / max(1.0, spec.inner_volume_cm3)))
        min_gap = min(
            max(0.0, (bl - max(u.raw_l for u in units_padded)) / 2.0),
            max(0.0, (bw - max(u.raw_w for u in units_padded)) / 2.0),
            max(0.0, (bh - max(u.raw_h for u in units_padded)) / 2.0),
        )
        avg_req = sum(required_pads) / len(required_pads)
        ratio = min(0.85, (min_gap / max(0.5, avg_req)) * 0.65 + free_ratio * 0.35)
        return max(0.15, round(ratio, 3))

    def _build_candidate_result(
        self,
        spec: ContainerSpec,
        box_count: int,
        raw_total_vol: float,
        padded_total_vol: float,
        cushion_ratio: float,
        placements: List[dict],
    ) -> dict:
        total_inner_vol = spec.inner_volume_cm3 * box_count
        total_outer_vol = spec.outer_volume_cm3 * box_count
        util_pct = min(0.95, raw_total_vol / total_inner_vol) if total_inner_vol > 0 else 0.0
        padded_util = (
            min(0.98, padded_total_vol / total_inner_vol) if total_inner_vol > 0 else 0.0
        )
        return {
            "box_id": spec.container_id,
            "box_count": box_count,
            "b_cost": spec.material_cost_inr * box_count,
            "box_vol": total_inner_vol,
            "outer_vol": total_outer_vol,
            "tare_wt": spec.tare_weight_kg * box_count,
            "utilization_pct": util_pct,
            "padded_utilization_pct": padded_util,
            "cushion_adequacy_ratio": cushion_ratio,
            "container_dims_cm": list(spec.inner_dims_cm),
            "spec": spec,
            "placements": placements,
        }

    def solve_3d_bin_packing(
        self,
        items: list,
        cost_matrix: Optional[dict] = None,
        require_cold_chain: bool = False,
        extra_containers: Optional[List[ContainerSpec]] = None,
    ) -> List[dict]:
        """Select feasible container(s) or multi-carton splits for *items*."""
        if not items:
            return [
                {
                    "box_id": "NO-BOX",
                    "box_count": 0,
                    "b_cost": 0.0,
                    "box_vol": 0.0,
                    "outer_vol": 0.0,
                    "tare_wt": 0.0,
                    "utilization_pct": 0.0,
                    "padded_utilization_pct": 0.0,
                    "cushion_adequacy_ratio": 1.0,
                    "container_dims_cm": [],
                    "spec": None,
                    "placements": [],
                }
            ]

        units_padded = self._expand_units(items, use_padding=True)
        units_raw = self._expand_units(items, use_padding=False)

        raw_total_vol = sum(u.raw_l * u.raw_w * u.raw_h for u in units_padded)
        padded_total_vol = sum(u.l * u.w * u.h for u in units_padded)
        total_weight = sum(u.weight_kg for u in units_padded)

        all_specs = list(self.ordered_specs)
        if extra_containers:
            existing_ids = {s.container_id for s in all_specs}
            for custom_spec in extra_containers:
                if custom_spec.container_id not in existing_ids:
                    all_specs.append(custom_spec)

        eligible_specs = sorted(
            [
                s
                for s in all_specs
                if (s.is_cold_chain if require_cold_chain else not s.is_cold_chain)
            ],
            key=lambda s: s.inner_volume_cm3,
        )

        valid_candidates: List[dict] = []
        single_fit_ids = set()

        # 1. Single-container padded pass (full ISTA cushioning)
        for spec in eligible_specs:
            if total_weight > spec.max_payload_kg:
                continue
            bl, bw, bh = spec.inner_dims_cm
            packed = self._cpsat_pack(units_padded, bl, bw, bh, bin_index=0)
            if packed is not None:
                single_fit_ids.add(spec.container_id)
                valid_candidates.append(
                    self._build_candidate_result(
                        spec=spec,
                        box_count=1,
                        raw_total_vol=raw_total_vol,
                        padded_total_vol=padded_total_vol,
                        cushion_ratio=1.0,
                        placements=packed,
                    )
                )

        # 2. Multi-carton parcel splitting (2..5 cartons) when a single carton doesn't fit
        split_ids = (
            ("CC-EPS-S", "CC-EPS-M", "CC-EPS-L")
            if require_cold_chain
            else ("C-SLIM-02", "C-MED-05", "C-FLAT-04", "C-LARGE-99", "C-HEAVY-5P")
        )
        if len(units_padded) >= 2:
            for cid in split_ids:
                if cid in single_fit_ids:
                    continue
                spec = self.catalog.get(cid)
                if spec is None:
                    continue
                multi_packed = self._try_multi_carton_split(units_padded, spec)
                if multi_packed is not None:
                    box_count = max(p["bin_index"] for p in multi_packed) + 1
                    valid_candidates.append(
                        self._build_candidate_result(
                            spec=spec,
                            box_count=box_count,
                            raw_total_vol=raw_total_vol,
                            padded_total_vol=padded_total_vol,
                            cushion_ratio=1.0,
                            placements=multi_packed,
                        )
                    )

        # 3. Fallback: unpadded pass if no parcel/cold_parcel container could fit with pads
        has_parcel_fit = any(
            c["spec"] is not None and c["spec"].tier_class in ("parcel", "cold_parcel")
            for c in valid_candidates
        )
        if not valid_candidates or not has_parcel_fit:
            for spec in eligible_specs:
                if spec.container_id in single_fit_ids:
                    continue
                if total_weight > spec.max_payload_kg:
                    continue
                bl, bw, bh = spec.inner_dims_cm
                packed = self._cpsat_pack(units_raw, bl, bw, bh, bin_index=0)
                if packed is not None:
                    cushion_ratio = self._compute_unpadded_cushion_ratio(
                        units_padded, spec
                    )
                    valid_candidates.append(
                        self._build_candidate_result(
                            spec=spec,
                            box_count=1,
                            raw_total_vol=raw_total_vol,
                            padded_total_vol=padded_total_vol,
                            cushion_ratio=cushion_ratio,
                            placements=packed,
                        )
                    )

        if valid_candidates:
            return sorted(
                valid_candidates,
                key=lambda x: (x["box_vol"], x["box_count"]),
            )

        # 4. Fallback: custom crate or multi-truck
        required_vol = padded_total_vol / 0.75
        custom_vol = required_vol * 1.1

        if custom_vol > 76_000_000:
            trucks_needed = int(custom_vol // 76_000_000) + 1
            custom_cost = trucks_needed * 20_000.0
            utilization = min(0.95, raw_total_vol / (trucks_needed * 76_000_000))
            padded_util = min(0.98, padded_total_vol / (trucks_needed * 76_000_000))
            return [
                {
                    "box_id": f"MULTI-TRUCK({trucks_needed}x40ft)",
                    "box_count": trucks_needed,
                    "b_cost": custom_cost,
                    "box_vol": trucks_needed * 76_000_000.0,
                    "outer_vol": trucks_needed * 86_000_000.0,
                    "tare_wt": 0.0,
                    "utilization_pct": utilization,
                    "padded_utilization_pct": padded_util,
                    "cushion_adequacy_ratio": 1.0,
                    "container_dims_cm": [1203.0, 235.0, 270.0],
                    "spec": None,
                    "placements": [],
                }
            ]

        custom_cost = min((custom_vol / 1000.0) * 22.5, 5000.0)
        utilization = min(0.95, raw_total_vol / custom_vol)
        padded_util = min(0.98, padded_total_vol / custom_vol)
        side = round(custom_vol ** (1.0 / 3.0), 1)
        return [
            {
                "box_id": "CUSTOM-CRATE" if not require_cold_chain else "CUSTOM-INSULATED-CRATE",
                "box_count": 1,
                "b_cost": custom_cost,
                "box_vol": custom_vol,
                "outer_vol": custom_vol * 1.12,
                "tare_wt": round(max(2.0, total_weight * 0.12), 2),
                "utilization_pct": utilization,
                "padded_utilization_pct": padded_util,
                "cushion_adequacy_ratio": 1.0,
                "container_dims_cm": [side, side, side],
                "spec": None,
                "placements": [],
            }
        ]
