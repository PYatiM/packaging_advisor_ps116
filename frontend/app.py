"""PackOptima Enterprise — Corporate Logistics & Packaging Control Tower."""

import json
import math
import os
import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

WORKSPACES = [
    "Order Studio & Dispatch",
    "3D Load Plan & Optimization",
    "Master SKU & Box Registry",
    "Audit Log & QA Telemetry",
    "Executive Analytics & ESG",
]

CATEGORY_OPTIONS = [
    "General",
    "Precision Electronics",
    "Perishable Goods",
    "Fine Art & Antiques",
    "Glass & Ceramics",
    "Industrial Machinery",
    "Textiles & Apparel",
    "Pharmaceuticals",
    "Automotive Parts",
    "Books & Media",
]

CATEGORY_BASE_FRAGILITY = {
    "General": 0.25,
    "Precision Electronics": 0.75,
    "Perishable Goods": 0.60,
    "Fine Art & Antiques": 0.90,
    "Glass & Ceramics": 0.85,
    "Industrial Machinery": 0.25,
    "Textiles & Apparel": 0.12,
    "Pharmaceuticals": 0.50,
    "Automotive Parts": 0.35,
    "Books & Media": 0.18,
}

PRIORITY_OPTIONS = {
    "balanced": "Balanced Landed Cost",
    "lowest_cost": "Lowest Direct Cost",
    "lowest_risk": "Maximum Protection",
    "eco_friendly": "Eco-Friendly Low Carbon",
}

HF_MODEL_CHOICES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct (Qwen Small 1.5B)",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct (Qwen Small 3B - Recommended)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct (Qwen 7B)",
    "PYatiM/packaging_advisor_ps116": "PYatiM/packaging_advisor_ps116 (Domain Fine-Tuned)",
}

GEMINI_MODEL_CHOICES = {
    "gemini-2.5-flash": "gemini-2.5-flash (Fast Balanced)",
    "gemini-2.5-pro": "gemini-2.5-pro (Deep Reasoning)",
}

SAMPLE_SCENARIOS = {
    "Mixed E-Commerce Cart": [
        {
            "sku_id": "ELEC-MBP-14",
            "product_category": "Precision Electronics",
            "length_cm": 35.0,
            "width_cm": 24.5,
            "height_cm": 4.5,
            "weight_kg": 1.85,
            "quantity": 1,
            "declared_value_inr": 68000.0,
            "is_liquid": False,
            "orientation_sensitive": False,
            "requires_cold_chain": False,
        },
        {
            "sku_id": "BOOK-CAT-15",
            "product_category": "Books & Media",
            "length_cm": 26.0,
            "width_cm": 19.0,
            "height_cm": 8.0,
            "weight_kg": 2.2,
            "quantity": 2,
            "declared_value_inr": 1500.0,
            "is_liquid": False,
            "orientation_sensitive": False,
            "requires_cold_chain": False,
        },
        {
            "sku_id": "LIQ-SOL-07",
            "product_category": "General",
            "length_cm": 11.0,
            "width_cm": 11.0,
            "height_cm": 24.0,
            "weight_kg": 1.35,
            "quantity": 1,
            "declared_value_inr": 750.0,
            "is_liquid": True,
            "orientation_sensitive": True,
            "requires_cold_chain": False,
        },
    ],
    "Cold-Chain Pharma & Perishables": [
        {
            "sku_id": "PHRM-INS-01",
            "product_category": "Pharmaceuticals",
            "length_cm": 16.0,
            "width_cm": 12.0,
            "height_cm": 9.0,
            "weight_kg": 0.85,
            "quantity": 1,
            "declared_value_inr": 12500.0,
            "is_liquid": True,
            "orientation_sensitive": True,
            "requires_cold_chain": True,
        },
        {
            "sku_id": "PERISH-MNG-05",
            "product_category": "Perishable Goods",
            "length_cm": 24.0,
            "width_cm": 18.0,
            "height_cm": 12.0,
            "weight_kg": 2.5,
            "quantity": 1,
            "declared_value_inr": 2400.0,
            "is_liquid": False,
            "orientation_sensitive": True,
            "requires_cold_chain": True,
        },
    ],
    "Fragile Glass + Heavy Machinery": [
        {
            "sku_id": "GLS-LAB-09",
            "product_category": "Glass & Ceramics",
            "length_cm": 22.0,
            "width_cm": 18.0,
            "height_cm": 20.0,
            "weight_kg": 2.1,
            "quantity": 1,
            "declared_value_inr": 4800.0,
            "is_liquid": False,
            "orientation_sensitive": True,
            "requires_cold_chain": False,
        },
        {
            "sku_id": "MECH-PMP-12",
            "product_category": "Industrial Machinery",
            "length_cm": 38.0,
            "width_cm": 28.0,
            "height_cm": 24.0,
            "weight_kg": 16.5,
            "quantity": 1,
            "declared_value_inr": 24000.0,
            "is_liquid": False,
            "orientation_sensitive": True,
            "requires_cold_chain": False,
        },
    ],
}


# ------------------------------------------------------------------
# Helper utilities & API client
# ------------------------------------------------------------------
def calculate_fragility(category: str, length: float, width: float, height: float, weight: float) -> float:
    base_frag = CATEGORY_BASE_FRAGILITY.get(category, 0.25)
    vol = max(1e-6, float(length) * float(width) * float(height))
    density = max(0.0, float(weight)) / (vol / 1000.0)
    if density > 0.0:
        base_frag -= math.log10(density + 1.0) * 0.04
    dims = sorted([max(0.1, float(length)), max(0.1, float(width)), max(0.1, float(height))])
    ratio = dims[2] / dims[0]
    if ratio > 1.0:
        base_frag += math.log(ratio) * 0.045
    return max(0.05, min(0.98, round(base_frag, 2)))


def format_volume(vol_cm3: float) -> str:
    if vol_cm3 >= 1_000_000:
        return f"{vol_cm3 / 1_000_000:.2f} m\u00b3"
    return f"{vol_cm3:,.0f} cm\u00b3"


def format_cost(cost: float) -> str:
    if cost >= 100_000:
        return f"Rs.{cost / 100_000:.2f}L"
    return f"Rs.{cost:,.0f}"


def sanitize_cart_rows(records: list) -> list:
    cleaned = []
    for row in records:
        sku = str(row.get("sku_id", "") or "").strip()
        if not sku:
            continue
        cat = row.get("product_category", "General")
        if cat not in CATEGORY_OPTIONS:
            cat = "General"
        l = max(0.5, float(row.get("length_cm") or 10.0))
        w = max(0.5, float(row.get("width_cm") or 10.0))
        h = max(0.5, float(row.get("height_cm") or 10.0))
        wt = max(0.05, float(row.get("weight_kg") or 1.0))
        qty = max(1, int(row.get("quantity") or 1))
        raw_val = row.get("declared_value_inr")
        declared = None
        if raw_val is not None and not pd.isna(raw_val) and float(raw_val) > 0:
            declared = round(float(raw_val), 2)

        cleaned.append(
            {
                "sku_id": sku,
                "product_category": cat,
                "length_cm": round(l, 2),
                "width_cm": round(w, 2),
                "height_cm": round(h, 2),
                "weight_kg": round(wt, 2),
                "quantity": qty,
                "declared_value_inr": declared,
                "fragility_score": calculate_fragility(cat, l, w, h, wt),
                "is_liquid": bool(row.get("is_liquid", False)),
                "orientation_sensitive": bool(row.get("orientation_sensitive", False)),
                "requires_cold_chain": bool(row.get("requires_cold_chain", False)),
            }
        )
    return cleaned


def api_get(path: str, params: dict | None = None, timeout: int = 12):
    try:
        res = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        if res.status_code == 200:
            return res.json(), None
        return None, f"HTTP {res.status_code}: {res.text}"
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, payload: dict, timeout: int = 60):
    try:
        res = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        if res.status_code in (200, 201):
            return res.json(), None
        return None, f"HTTP {res.status_code}: {res.text}"
    except Exception as exc:
        return None, str(exc)


def api_patch(path: str, payload: dict, timeout: int = 15):
    try:
        res = requests.patch(f"{API_URL}{path}", json=payload, timeout=timeout)
        if res.status_code == 200:
            return res.json(), None
        return None, f"HTTP {res.status_code}: {res.text}"
    except Exception as exc:
        return None, str(exc)


def api_delete(path: str, timeout: int = 15):
    try:
        res = requests.delete(f"{API_URL}{path}", timeout=timeout)
        if res.status_code == 200:
            return res.json(), None
        return None, f"HTTP {res.status_code}: {res.text}"
    except Exception as exc:
        return None, str(exc)


def render_badge_strip(badges: list, is_dark: bool) -> None:
    if not badges:
        return
    color_map = {
        "COLD CHAIN": ("#0284c7", "#e0f2fe") if not is_dark else ("#38bdf8", "rgba(56,189,248,0.16)"),
        "LIQUID": ("#b45309", "#fef3c7") if not is_dark else ("#fbbf24", "rgba(251,191,36,0.16)"),
        "ESD": ("#7c3aed", "#ede9fe") if not is_dark else ("#a78bfa", "rgba(167,139,250,0.16)"),
        "ISTA": ("#059669", "#d1fae5") if not is_dark else ("#34d399", "rgba(52,211,153,0.16)"),
        "HEAVY": ("#dc2626", "#fee2e2") if not is_dark else ("#f87171", "rgba(248,113,113,0.16)"),
        "THIS WAY": ("#0f766e", "#ccfbf1") if not is_dark else ("#2dd4bf", "rgba(45,212,191,0.16)"),
    }
    pills = []
    for b in badges:
        fg, bg = ("#3b82f6", "rgba(59,130,246,0.15)")
        for prefix, pair in color_map.items():
            if prefix in b.upper():
                fg, bg = pair
                break
        pills.append(
            f"<span style='display:inline-block;padding:3px 9px;margin:2px 6px 2px 0;"
            f"border-radius:999px;border:1px solid {fg};background:{bg};color:{fg};"
            f"font-size:11px;font-weight:700;letter-spacing:0.3px;'>{b}</span>"
        )
    st.markdown(f"<div style='margin:6px 0 10px 0;'>{''.join(pills)}</div>", unsafe_allow_html=True)


def render_3d_packing_viewer(
    placements: list,
    container_dims: list,
    box_count: int,
    carton_id: str,
    theme_mode: str,
) -> None:
    """Interactive HTML5 3D load planner with Exploded View & ISTA Cushion Shell."""
    if not placements or len(container_dims) != 3:
        st.info("3D spatial coordinates are unavailable for custom/multi-truck fallback or historical records without saved coordinates.")
        return

    is_dark = theme_mode == "Corporate Dark"
    payload = json.dumps(
        {
            "dims": container_dims,
            "placements": placements,
            "boxCount": box_count,
            "cartonId": carton_id,
            "dark": is_dark,
        }
    )

    html = f"""
    <div id="packer-root" style="font-family: Inter, system-ui, sans-serif; color: {'#e2e8f0' if is_dark else '#0f172a'};">
      <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:10px; font-size:12.5px; background:{'#111827' if is_dark else '#f1f5f9'}; padding:8px 12px; border-radius:8px; border:1px solid {'#1f2937' if is_dark else '#cbd5e1'};">
        <span><b>Container:</b> {carton_id} ({container_dims[0]:.0f} x {container_dims[1]:.0f} x {container_dims[2]:.0f} cm)</span>
        <span id="bin-select-wrap" style="display:{'inline-flex' if box_count > 1 else 'none'}; align-items:center; gap:6px;">
          <label for="bin-select"><b>Carton:</b></label>
          <select id="bin-select" style="padding:3px 8px; border-radius:6px; border:1px solid #475569; background:{'#1e293b' if is_dark else '#ffffff'}; color:inherit;"></select>
        </span>
        <label style="display:inline-flex; align-items:center; gap:6px;">
          Zoom <input id="zoom-slider" type="range" min="0.6" max="1.9" step="0.05" value="1.05" />
        </label>
        <label style="display:inline-flex; align-items:center; gap:6px;">
          Explode <input id="explode-slider" type="range" min="0" max="1" step="0.05" value="0" />
        </label>
        <label style="display:inline-flex; align-items:center; gap:5px; cursor:pointer;">
          <input id="cushion-toggle" type="checkbox" checked /> ISTA Cushion Shell
        </label>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:flex-start;">
        <canvas id="pack-canvas" width="580" height="400"
          style="border-radius:10px; border:1px solid {'#1f2937' if is_dark else '#cbd5e1'};
                 background:{'#090f1d' if is_dark else '#f8fafc'}; cursor:grab;"></canvas>
        <div id="pack-legend" style="flex:1; min-width:220px; max-height:400px; overflow-y:auto; font-size:12px;"></div>
      </div>
    </div>
    <script>
    (function() {{
      const DATA = {payload};
      const canvas = document.getElementById('pack-canvas');
      const ctx = canvas.getContext('2d');
      const legendEl = document.getElementById('pack-legend');
      const binSelect = document.getElementById('bin-select');
      const zoomSlider = document.getElementById('zoom-slider');
      const explodeSlider = document.getElementById('explode-slider');
      const cushionToggle = document.getElementById('cushion-toggle');

      const palette = [
        '#38bdf8', '#fb923c', '#34d399', '#a78bfa',
        '#f472b6', '#facc15', '#2dd4bf', '#f87171'
      ];

      let yaw = -0.68;
      let pitch = 0.44;
      let zoom = 1.05;
      let explode = 0.0;
      let showCushion = true;
      let activeBin = 0;
      let dragging = false;
      let lastX = 0, lastY = 0;

      if (DATA.boxCount > 1) {{
        for (let b = 0; b < DATA.boxCount; b++) {{
          const opt = document.createElement('option');
          opt.value = b;
          opt.textContent = 'Box ' + (b + 1) + ' of ' + DATA.boxCount;
          binSelect.appendChild(opt);
        }}
        binSelect.addEventListener('change', (e) => {{
          activeBin = parseInt(e.target.value, 10) || 0;
          draw();
        }});
      }}

      zoomSlider.addEventListener('input', (e) => {{ zoom = parseFloat(e.target.value); draw(); }});
      explodeSlider.addEventListener('input', (e) => {{ explode = parseFloat(e.target.value); draw(); }});
      cushionToggle.addEventListener('change', (e) => {{ showCushion = e.target.checked; draw(); }});

      canvas.addEventListener('mousedown', (e) => {{
        dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.style.cursor = 'grabbing';
      }});
      window.addEventListener('mouseup', () => {{ dragging = false; canvas.style.cursor = 'grab'; }});
      window.addEventListener('mousemove', (e) => {{
        if (!dragging) return;
        yaw += (e.clientX - lastX) * 0.009;
        pitch = Math.max(-1.25, Math.min(1.25, pitch + (e.clientY - lastY) * 0.009));
        lastX = e.clientX; lastY = e.clientY;
        draw();
      }});

      function shadeHex(hex, factor) {{
        const c = parseInt(hex.slice(1), 16);
        const r = Math.min(255, Math.max(0, Math.round(((c >> 16) & 255) * factor)));
        const g = Math.min(255, Math.max(0, Math.round(((c >> 8) & 255) * factor)));
        const b = Math.min(255, Math.max(0, Math.round((c & 255) * factor)));
        return `rgb(${{r}},${{g}},${{b}})`;
      }}

      function project(px, py, pz, L, W, H, scale) {{
        const cx = px - L / 2;
        const cy = py - W / 2;
        const cz = pz - H / 2;
        const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
        const x1 = cx * cosY - cy * sinY;
        const y1 = cx * sinY + cy * cosY;
        const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
        const y2 = y1 * cosP - cz * sinP;
        const z2 = y1 * sinP + cz * cosP;
        const persp = 680 / (680 + y2 * scale * 0.45);
        return {{ sx: canvas.width / 2 + x1 * scale * persp, sy: canvas.height / 2 - z2 * scale * persp, depth: y2 }};
      }}

      function buildCuboidFaces(x, y, z, dx, dy, dz, color, alpha, L, W, H, scale, strokeCol) {{
        const pts = [
          [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
          [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ].map(p => project(p[0], p[1], p[2], L, W, H, scale));
        const defs = [
          {{ idx: [4,5,6,7], shade: 1.06 }},
          {{ idx: [0,1,2,3], shade: 0.58 }},
          {{ idx: [0,1,5,4], shade: 0.86 }},
          {{ idx: [2,3,7,6], shade: 0.74 }},
          {{ idx: [1,2,6,5], shade: 0.94 }},
          {{ idx: [3,0,4,7], shade: 0.68 }}
        ];
        return defs.map(f => ({{
          poly: f.idx.map(i => pts[i]),
          depth: f.idx.reduce((acc, i) => acc + pts[i].depth, 0) / 4,
          fill: shadeHex(color, f.shade),
          alpha: alpha,
          stroke: strokeCol
        }}));
      }}

      function draw() {{
        const [L, W, H] = DATA.dims;
        const maxSpan = Math.max(L, W, H, 1);
        const scale = (205 / maxSpan) * zoom * (1.0 - explode * 0.22);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const binItems = DATA.placements.filter(p => (p.bin_index || 0) === activeBin);
        legendEl.innerHTML = '<div style="font-weight:700; margin-bottom:6px;">Packed Units (' + binItems.length + ')</div>' +
          binItems.map((it, idx) => {{
            const col = palette[idx % palette.length];
            const padTag = it.pad_cm > 0 ? ` | ISTA pad: ${{it.pad_cm}}cm` : '';
            return `<div style="display:flex; gap:8px; align-items:flex-start; margin-bottom:7px; padding:7px; border-radius:6px; background:${{DATA.dark ? '#111827' : '#f1f5f9'}}; border:1px solid ${{DATA.dark ? '#1f2937' : '#e2e8f0'}};">
              <span style="width:12px; height:12px; border-radius:3px; background:${{col}}; flex-shrink:0; margin-top:3px;"></span>
              <div>
                <div style="font-weight:600;">${{it.sku_id}}</div>
                <div style="opacity:0.85;">${{it.dx_cm}} x ${{it.dy_cm}} x ${{it.dz_cm}} cm${{padTag}}</div>
                <div style="opacity:0.65;">Pos: (${{it.x_cm}}, ${{it.y_cm}}, ${{it.z_cm}}) | ${{it.weight_kg}} kg</div>
              </div>
            </div>`;
          }}).join('');

        const corners = [
          [0,0,0], [L,0,0], [L,W,0], [0,W,0],
          [0,0,H], [L,0,H], [L,W,H], [0,W,H]
        ].map(p => project(p[0], p[1], p[2], L, W, H, scale));
        const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

        ctx.lineWidth = 1.4;
        ctx.strokeStyle = DATA.dark ? 'rgba(148,163,184,0.48)' : 'rgba(51,65,85,0.48)';
        ctx.setLineDash([5, 4]);
        edges.forEach(([a, b]) => {{
          ctx.beginPath(); ctx.moveTo(corners[a].sx, corners[a].sy); ctx.lineTo(corners[b].sx, corners[b].sy); ctx.stroke();
        }});
        ctx.setLineDash([]);

        let faces = [];
        binItems.forEach((it, idx) => {{
          const col = palette[idx % palette.length];
          const cx = it.x_cm + it.dx_cm / 2 - L / 2;
          const cy = it.y_cm + it.dy_cm / 2 - W / 2;
          const cz = it.z_cm + it.dz_cm / 2 - H / 2;
          const ox = it.x_cm + cx * explode * 0.85;
          const oy = it.y_cm + cy * explode * 0.85;
          const oz = it.z_cm + cz * explode * 0.85 + explode * idx * (H * 0.08);
          const pad = it.pad_cm || 0;

          if (showCushion && pad > 0 && it.dx_cm > 2 * pad && it.dy_cm > 2 * pad && it.dz_cm > 2 * pad) {{
            // Inner solid core
            faces = faces.concat(buildCuboidFaces(
              ox + pad, oy + pad, oz + pad,
              it.dx_cm - 2 * pad, it.dy_cm - 2 * pad, it.dz_cm - 2 * pad,
              col, 0.94, L, W, H, scale,
              DATA.dark ? 'rgba(15,23,42,0.85)' : 'rgba(15,23,42,0.65)'
            ));
            // Outer translucent ISTA cushion envelope
            faces = faces.concat(buildCuboidFaces(
              ox, oy, oz, it.dx_cm, it.dy_cm, it.dz_cm,
              '#94a3b8', 0.18, L, W, H, scale,
              'rgba(56,189,248,0.55)'
            ));
          }} else {{
            faces = faces.concat(buildCuboidFaces(
              ox, oy, oz, it.dx_cm, it.dy_cm, it.dz_cm,
              col, 0.90, L, W, H, scale,
              DATA.dark ? 'rgba(15,23,42,0.75)' : 'rgba(15,23,42,0.45)'
            ));
          }}
        }});
        faces.sort((a, b) => b.depth - a.depth);

        faces.forEach(f => {{
          ctx.beginPath();
          ctx.moveTo(f.poly[0].sx, f.poly[0].sy);
          for (let i = 1; i < f.poly.length; i++) ctx.lineTo(f.poly[i].sx, f.poly[i].sy);
          ctx.closePath();
          ctx.fillStyle = f.fill;
          ctx.globalAlpha = f.alpha;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.lineWidth = 1.0;
          ctx.strokeStyle = f.stroke;
          ctx.stroke();
        }});

        ctx.fillStyle = DATA.dark ? '#94a3b8' : '#334155';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(`L: ${{L}}cm`, (corners[0].sx + corners[1].sx)/2 - 18, (corners[0].sy + corners[1].sy)/2 + 14);
        ctx.fillText(`W: ${{W}}cm`, (corners[1].sx + corners[2].sx)/2 + 4, (corners[1].sy + corners[2].sy)/2 + 12);
        ctx.fillText(`H: ${{H}}cm`, corners[4].sx - 24, (corners[0].sy + corners[4].sy)/2);
      }}

      draw();
    }})();
    </script>
    """
    components.html(html, height=470, scrolling=False)


# ------------------------------------------------------------------
# Page setup & state initialization
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PackOptima Enterprise | Packaging Control Tower",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "active_workspace": WORKSPACES[0],
    "theme_mode": "Corporate Dark",
    "cart": [],
    "customer_or_po": "PO-ENT-2026-01",
    "source_pin": "110001",
    "destination_pin": "560001",
    "optimization_priority": "balanced",
    "use_genai": False,
    "model_provider": "Gemini",
    "selected_model_name": "gemini-2.5-flash",
    "custom_hf_model": "Qwen/Qwen2.5-3B-Instruct",
    "last_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

is_dark = st.session_state.theme_mode == "Corporate Dark"
page_bg = "#0B1120" if is_dark else "#F8FAFC"
panel_bg = "#111827" if is_dark else "#FFFFFF"
soft_bg = "#1E293B" if is_dark else "#F1F5F9"
border_col = "#1F2937" if is_dark else "#E2E8F0"
text_col = "#F8FAFC" if is_dark else "#0F172A"
muted_col = "#94A3B8" if is_dark else "#475569"
accent = "#38BDF8" if is_dark else "#2563EB"
positive = "#10B981"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background: {page_bg}; color: {text_col}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {panel_bg}; border-right: 1px solid {border_col}; }}
    [data-testid="stMetric"] {{
        background: {panel_bg};
        border: 1px solid {border_col};
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }}
    [data-testid="stMetricValue"] {{ color: {accent} !important; font-weight: 800; }}
    .corp-card {{
        background: {panel_bg};
        border: 1px solid {border_col};
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }}
    .corp-banner {{
        background: linear-gradient(135deg, {'#0F172A 0%, #1E293B 100%' if is_dark else '#EFF6FF 0%, #FFFFFF 100%'});
        border: 1px solid {border_col};
        border-left: 4px solid {accent};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }}
    .stButton button[kind="primary"] {{
        background: {accent} !important;
        border: 1px solid {accent} !important;
        color: {'#0B1120' if is_dark else '#FFFFFF'} !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }}
    h1, h2, h3, h4, h5, p, label, span {{ color: {text_col}; }}
    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def run_cart_optimization(items: list, src_pin: str, dst_pin: str, po_ref: str, priority: str):
    payload = {
        "items": items,
        "source_pin": src_pin,
        "destination_pin": dst_pin,
        "use_genai": st.session_state.use_genai,
        "model_provider": st.session_state.model_provider,
        "model_name": st.session_state.selected_model_name,
        "optimization_priority": priority,
        "customer_or_po": po_ref,
        "persist_history": True,
    }
    return api_post("/recommend", payload, timeout=60)


# ------------------------------------------------------------------
# Sidebar Navigation Rail
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:10px 12px; border-radius:10px; background:{soft_bg}; border:1px solid {border_col}; margin-bottom:14px;">
          <div style="font-size:18px; font-weight:800; color:{accent}; letter-spacing:-0.3px;">PackOptima™ Enterprise</div>
          <div style="font-size:11.5px; color:{muted_col}; margin-top:2px;">IS 2771 • ISTA 3A • Zone A–E Control Tower</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    health_data, health_err = api_get("/health", timeout=4)
    if health_data:
        ml_badge = "ML Ready" if health_data.get("ml_model_loaded") else "Heuristic"
        db_badge = "DB Live" if health_data.get("database_connected") else "No DB"
        st.markdown(
            f"<div style='font-size:11.5px; color:{positive}; font-weight:600; margin-bottom:10px;'>"
            f"● API Connected &nbsp;|&nbsp; {db_badge} &nbsp;|&nbsp; {ml_badge}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error(f"API Offline ({API_URL}): {health_err}")

    selected_ws = st.radio(
        "Control Tower Navigation",
        WORKSPACES,
        index=WORKSPACES.index(st.session_state.active_workspace),
    )
    if selected_ws != st.session_state.active_workspace:
        st.session_state.active_workspace = selected_ws
        st.rerun()

    st.divider()
    st.caption("GLOBAL PREFERENCES")
    theme_choice = st.radio(
        "Theme",
        ["Corporate Dark", "Executive Light"],
        index=0 if is_dark else 1,
        horizontal=True,
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()

    st.session_state.use_genai = st.toggle(
        "GenAI Post-Hoc Narrative",
        value=st.session_state.use_genai,
        help="Generates natural-language SOP insights without altering packing or cost math.",
    )
    st.session_state.model_provider = st.selectbox(
        "LLM Provider",
        ["Gemini", "Hugging Face"],
        index=0 if st.session_state.model_provider == "Gemini" else 1,
    )
    if st.session_state.model_provider == "Hugging Face":
        hf_keys = list(HF_MODEL_CHOICES.keys()) + ["__custom__"]
        cur_hf = st.session_state.selected_model_name
        default_idx = hf_keys.index(cur_hf) if cur_hf in HF_MODEL_CHOICES else (
            hf_keys.index("Qwen/Qwen2.5-3B-Instruct")
            if cur_hf in GEMINI_MODEL_CHOICES
            else len(hf_keys) - 1
        )
        picked_hf = st.selectbox(
            "Hugging Face Model",
            hf_keys,
            index=default_idx,
            format_func=lambda k: (
                "Custom HF Repo ID..." if k == "__custom__" else HF_MODEL_CHOICES[k]
            ),
        )
        if picked_hf == "__custom__":
            st.session_state.custom_hf_model = st.text_input(
                "HF Model Repo ID",
                value=st.session_state.custom_hf_model,
            )
            st.session_state.selected_model_name = (
                st.session_state.custom_hf_model.strip() or "Qwen/Qwen2.5-3B-Instruct"
            )
        else:
            st.session_state.selected_model_name = picked_hf

        if health_data and health_data.get("huggingface_configured"):
            st.caption("● HF Token Ready (`HUGGINGFACE_API_KEY`)")
        else:
            st.caption("○ Set `HUGGINGFACE_API_KEY` in `.env` (fallback active)")
    else:
        gem_keys = list(GEMINI_MODEL_CHOICES.keys())
        cur_gem = st.session_state.selected_model_name
        g_idx = gem_keys.index(cur_gem) if cur_gem in gem_keys else 0
        st.session_state.selected_model_name = st.selectbox(
            "Gemini Model",
            gem_keys,
            index=g_idx,
            format_func=lambda k: GEMINI_MODEL_CHOICES[k],
        )
        if health_data and health_data.get("gemini_configured"):
            st.caption("● Gemini Key Ready (`GEMINI_API_KEY`)")
        else:
            st.caption("○ Set `GEMINI_API_KEY` in `.env` (fallback active)")


# ------------------------------------------------------------------
# WORKSPACE 1: Order Studio & Dispatch
# ------------------------------------------------------------------
if st.session_state.active_workspace == "Order Studio & Dispatch":
    st.markdown(
        """
        <div class="corp-banner">
          <h2 style="margin:0;">Order Studio & Dispatch Planning</h2>
          <p style="margin:4px 0 0 0; opacity:0.82;">Build multi-SKU orders from the enterprise database or ad-hoc inputs, select an objective, and generate a 3D load plan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        c_po, c_src, c_dst, c_pri, c_go = st.columns([1.4, 0.9, 0.9, 1.5, 1.3])
        st.session_state.customer_or_po = c_po.text_input(
            "Customer / PO Ref", value=st.session_state.customer_or_po
        )
        st.session_state.source_pin = c_src.text_input(
            "Origin PIN", value=st.session_state.source_pin, max_chars=6
        )
        st.session_state.destination_pin = c_dst.text_input(
            "Destination PIN", value=st.session_state.destination_pin, max_chars=6
        )
        st.session_state.optimization_priority = c_pri.selectbox(
            "Optimization Objective",
            list(PRIORITY_OPTIONS.keys()),
            index=list(PRIORITY_OPTIONS.keys()).index(st.session_state.optimization_priority),
            format_func=lambda k: PRIORITY_OPTIONS[k],
        )
        c_go.write("")
        c_go.write("")
        if c_go.button("Run Optimization & Open Load Plan", type="primary", use_container_width=True):
            if not st.session_state.cart:
                st.warning("Add at least one SKU to the order manifest below.")
            elif not (
                st.session_state.source_pin.isdigit()
                and len(st.session_state.source_pin) == 6
                and st.session_state.destination_pin.isdigit()
                and len(st.session_state.destination_pin) == 6
            ):
                st.error("Both Origin and Destination PINs must be 6 digits.")
            else:
                with st.spinner("Solving 3D CP-SAT packing, Zone A-E routing & calibrated ML scoring..."):
                    data, err = run_cart_optimization(
                        st.session_state.cart,
                        st.session_state.source_pin,
                        st.session_state.destination_pin,
                        st.session_state.customer_or_po,
                        st.session_state.optimization_priority,
                    )
                if err:
                    st.error(f"Optimization failed: {err}")
                else:
                    st.session_state.last_result = data
                    st.session_state.active_workspace = "3D Load Plan & Optimization"
                    st.rerun()

    col_builder, col_manifest = st.columns([1.05, 1.55])

    with col_builder:
        tab_db, tab_custom, tab_import = st.tabs(
            ["Master SKU Catalog", "Custom Ad-Hoc SKU", "Presets & Bulk Import"]
        )

        with tab_db:
            @st.fragment
            def master_sku_picker():
                skus_data, skus_err = api_get("/skus")
                if skus_err or not skus_data:
                    st.warning("Could not load Master SKUs from DB. Use Custom SKU or Presets.")
                    return
                sku_search = st.text_input("Filter Catalog SKUs", placeholder="Search code or name...")
                filtered = [
                    s
                    for s in skus_data
                    if not sku_search.strip()
                    or sku_search.lower() in s["sku_code"].lower()
                    or sku_search.lower() in s["name"].lower()
                ]
                if not filtered:
                    st.info("No matching Master SKUs.")
                    return
                labels = {
                    f"{s['sku_code']} — {s['name']} ({s['product_category']})": s
                    for s in filtered
                }
                picked_label = st.selectbox("Select SKU", list(labels.keys()))
                picked = labels[picked_label]
                st.caption(
                    f"Dims: {picked['length_cm']}x{picked['width_cm']}x{picked['height_cm']} cm | "
                    f"Wt: {picked['weight_kg']} kg | Value: Rs.{picked['declared_value_inr']:,.0f} | "
                    f"Fragility: {picked['fragility_score']}"
                )
                qty_pick = st.number_input("Order Quantity", min_value=1, max_value=200, value=1, step=1)
                if st.button("Add Master SKU to Order", type="primary", use_container_width=True):
                    st.session_state.cart.append(
                        {
                            "sku_id": picked["sku_code"],
                            "product_category": picked["product_category"],
                            "length_cm": picked["length_cm"],
                            "width_cm": picked["width_cm"],
                            "height_cm": picked["height_cm"],
                            "weight_kg": picked["weight_kg"],
                            "quantity": int(qty_pick),
                            "declared_value_inr": picked["declared_value_inr"],
                            "fragility_score": picked["fragility_score"],
                            "is_liquid": picked["is_liquid"],
                            "orientation_sensitive": picked["orientation_sensitive"],
                            "requires_cold_chain": picked["requires_cold_chain"],
                        }
                    )
                    st.rerun()

            master_sku_picker()

        with tab_custom:
            sku_code = st.text_input("SKU Code", placeholder="SKU-2045")
            sku_cat = st.selectbox("Product Category", CATEGORY_OPTIONS)
            d1, d2 = st.columns(2)
            l_cm = d1.number_input("Length (cm)", min_value=1.0, value=20.0, step=1.0)
            w_cm = d2.number_input("Width (cm)", min_value=1.0, value=15.0, step=1.0)
            h_cm = d1.number_input("Height (cm)", min_value=1.0, value=10.0, step=1.0)
            wt_kg = d2.number_input("Unit Weight (kg)", min_value=0.1, value=1.2, step=0.1)
            q_col, v_col = st.columns(2)
            qty_custom = q_col.number_input("Qty", min_value=1, max_value=200, value=1, step=1)
            val_custom = v_col.number_input("Declared Value (Rs./unit)", min_value=0.0, value=0.0, step=100.0)
            f1, f2, f3 = st.columns(3)
            is_liq = f1.checkbox("Liquid")
            is_ori = f2.checkbox("This Way Up")
            is_cold = f3.checkbox("Cold-Chain", value=(sku_cat == "Perishable Goods"))
            save_to_db = st.checkbox("Also save to Master SKU Catalog (DB)")

            if st.button("Add Custom SKU", use_container_width=True):
                if not sku_code.strip():
                    st.error("SKU Code is required.")
                else:
                    frag = calculate_fragility(sku_cat, l_cm, w_cm, h_cm, wt_kg)
                    item_dict = {
                        "sku_id": sku_code.strip().upper(),
                        "product_category": sku_cat,
                        "length_cm": float(l_cm),
                        "width_cm": float(w_cm),
                        "height_cm": float(h_cm),
                        "weight_kg": float(wt_kg),
                        "quantity": int(qty_custom),
                        "declared_value_inr": float(val_custom) if val_custom > 0 else None,
                        "fragility_score": frag,
                        "is_liquid": is_liq,
                        "orientation_sensitive": is_ori,
                        "requires_cold_chain": is_cold or (sku_cat == "Perishable Goods"),
                    }
                    st.session_state.cart.append(item_dict)
                    if save_to_db:
                        api_post(
                            "/skus",
                            {
                                "sku_code": item_dict["sku_id"],
                                "name": item_dict["sku_id"],
                                "product_category": sku_cat,
                                "length_cm": l_cm,
                                "width_cm": w_cm,
                                "height_cm": h_cm,
                                "weight_kg": wt_kg,
                                "fragility_score": frag,
                                "declared_value_inr": item_dict["declared_value_inr"],
                                "is_liquid": is_liq,
                                "orientation_sensitive": is_ori,
                                "requires_cold_chain": item_dict["requires_cold_chain"],
                            },
                        )
                    st.rerun()

        with tab_import:
            scenario_name = st.selectbox("Preset Scenario", list(SAMPLE_SCENARIOS.keys()))
            if st.button("Load Preset Scenario", use_container_width=True):
                st.session_state.cart = sanitize_cart_rows(SAMPLE_SCENARIOS[scenario_name])
                st.rerun()

            st.divider()
            csv_file = st.file_uploader("Upload CSV Order Manifest", type=["csv"])
            if csv_file is not None and st.button("Import CSV Rows", use_container_width=True):
                try:
                    csv_df = pd.read_csv(csv_file)
                    st.session_state.cart.extend(sanitize_cart_rows(csv_df.to_dict("records")))
                    st.rerun()
                except Exception as exc:
                    st.error(f"CSV parse error: {exc}")

            raw_json = st.text_area("Paste JSON Item Array", height=90)
            if st.button("Import JSON Array", use_container_width=True):
                try:
                    loaded = json.loads(raw_json)
                    if not isinstance(loaded, list):
                        raise ValueError("JSON root must be an array.")
                    st.session_state.cart.extend(sanitize_cart_rows(loaded))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Invalid JSON: {exc}")

    with col_manifest:
        st.subheader("Active Order Manifest")
        units = sum(int(i.get("quantity", 1)) for i in st.session_state.cart)
        tot_wt = sum(i["weight_kg"] * int(i.get("quantity", 1)) for i in st.session_state.cart)
        tot_vol = sum(
            i["length_cm"] * i["width_cm"] * i["height_cm"] * int(i.get("quantity", 1))
            for i in st.session_state.cart
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Line Items / Units", f"{len(st.session_state.cart)} / {units}")
        m2.metric("Net Payload Weight", f"{tot_wt:,.1f} kg")
        m3.metric("Net Cargo Volume", format_volume(tot_vol))

        if st.session_state.cart:
            df_cart = pd.DataFrame(st.session_state.cart)
            edited = st.data_editor(
                df_cart,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                disabled=["fragility_score"],
                column_config={
                    "product_category": st.column_config.SelectboxColumn(
                        "Category", options=CATEGORY_OPTIONS, required=True
                    ),
                    "quantity": st.column_config.NumberColumn("Qty", min_value=1, max_value=200, step=1),
                },
            )
            st.session_state.cart = sanitize_cart_rows(edited.to_dict("records"))

            b1, b2 = st.columns(2)
            if b1.button("Remove Last Line", use_container_width=True):
                st.session_state.cart.pop()
                st.rerun()
            if b2.button("Clear Manifest", use_container_width=True):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("Manifest empty. Pick a Master SKU on the left or load a Preset Scenario to test right away.")


# ------------------------------------------------------------------
# WORKSPACE 2: 3D Load Plan & Optimization
# ------------------------------------------------------------------
elif st.session_state.active_workspace == "3D Load Plan & Optimization":
    data = st.session_state.last_result
    if not data:
        st.info("No active optimization loaded yet. Build an order or run a quick demo below.")
        if st.button("Load & Run Demo Order (Delhi -> Bengaluru)", type="primary"):
            demo_items = sanitize_cart_rows(SAMPLE_SCENARIOS["Mixed E-Commerce Cart"])
            st.session_state.cart = demo_items
            res, err = run_cart_optimization(
                demo_items, "110001", "560001", "PO-DEMO-2026", "balanced"
            )
            if err:
                st.error(err)
            else:
                st.session_state.last_result = res
                st.rerun()
    else:
        order_ref = data.get("order_ref", "")
        st.markdown(
            f"""
            <div class="corp-banner">
              <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                  <h2 style="margin:0;">3D Load Plan & Dispatch Decision</h2>
                  <p style="margin:4px 0 0 0; opacity:0.85;">
                    <b>Order Ref:</b> {order_ref or 'Preview'} &nbsp;|&nbsp;
                    <b>Lane:</b> {data.get('source_zone')} &rarr; {data.get('destination_zone')} ({data.get('shipping_zone_tier')})
                  </p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if data.get("compatibility_warning"):
            st.warning(data["compatibility_warning"])

        shipments = data.get("shipments", [])
        avg_risk = (
            sum(s["damage_risk_probability"] for s in shipments) / len(shipments) * 100.0
            if shipments
            else 0.0
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Total Direct Cost",
            format_cost(data["total_cart_cost"]),
            delta=f"-{format_cost(data.get('cost_savings_inr', 0.0))} ({data.get('cost_savings_pct', 0.0):.1f}% vs oversize)",
            delta_color="normal",
        )
        k2.metric(
            "Effective Landed Cost",
            format_cost(data.get("total_effective_landed_cost", data["total_cart_cost"])),
            delta=f"Risk Exposure: {format_cost(data.get('total_expected_damage_cost', 0.0))}",
            delta_color="off",
        )
        k3.metric(
            "Mean Damage Risk",
            f"{avg_risk:.1f}%",
            delta=f"{len(shipments)} compatible shipment(s)",
            delta_color="off",
        )
        k4.metric(
            "Scope 3 Carbon (CO2e)",
            f"{data.get('total_co2_kg', 0.0):.2f} kg",
            delta=f"-{data.get('co2_saved_kg', 0.0):.2f} kg saved vs oversize",
            delta_color="normal",
        )

        # Export & Dispatch Manifest action bar
        if order_ref:
            manifest, _ = api_get(f"/orders/{order_ref}/manifest")
            if manifest:
                e1, e2, e3 = st.columns(3)
                e1.download_button(
                    "Download Printable HTML Manifest",
                    data=manifest["html_manifest"],
                    file_name=f"{order_ref}_manifest.html",
                    mime="text/html",
                    use_container_width=True,
                )
                e2.download_button(
                    "Download Markdown SOP Sheet",
                    data=manifest["markdown_manifest"],
                    file_name=f"{order_ref}_manifest.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                e3.download_button(
                    "Download JSON Payload",
                    data=json.dumps(data, indent=2),
                    file_name=f"{order_ref}_loadplan.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.divider()
        for idx, sh in enumerate(shipments):
            grp = sh.get("group_label", "general").replace("_", " ").upper()
            header = (
                f"Shipment #{idx + 1} [{grp}] — {sh['recommended_carton_id']} | "
                f"{format_cost(sh['total_estimated_cost'])} | "
                f"{sh.get('transit_mode', '')} ({sh.get('estimated_transit_days', '')})"
            )
            with st.expander(header, expanded=True):
                render_badge_strip(sh.get("handling_badges", []), is_dark)
                st.info(
                    f"**Board & Spec:** {sh.get('container_spec_summary', '')} &nbsp;|&nbsp; "
                    f"**Chargeable Wt:** {sh.get('chargeable_weight_kg', 0):.1f} kg "
                    f"(Gross: {sh.get('actual_gross_weight_kg', 0):.1f} kg, Vol: {sh.get('volumetric_weight_kg', 0):.1f} kg) &nbsp;|&nbsp; "
                    f"**Utilization:** {sh.get('net_utilization_pct', 0):.1f}% net / {sh.get('padded_utilization_pct', 0):.1f}% padded &nbsp;|&nbsp; "
                    f"**CO2e:** {sh.get('co2_emissions_kg', 0):.2f} kg"
                )

                t_3d, t_alts, t_sop = st.tabs(
                    ["Interactive 3D Load Planner", "Cost, Carbon & Risk Alternatives", "Packing SOP & AI Rationale"]
                )
                with t_3d:
                    render_3d_packing_viewer(
                        sh.get("placements", []),
                        sh.get("container_inner_dims_cm", []),
                        sh.get("box_count", 1),
                        sh["recommended_carton_id"],
                        st.session_state.theme_mode,
                    )
                with t_alts:
                    alts = sh.get("alternatives", [])
                    if alts:
                        alt_cols = st.columns(len(alts))
                        for a_col, alt_item in zip(alt_cols, alts):
                            with a_col:
                                cb_html = alt_item["cost_breakdown"].replace(" | ", "<br>• ")
                                st.markdown(
                                    f"""
                                    <div class="corp-card">
                                      <div style="color:{accent}; font-weight:700; font-size:13px;">{alt_item['tier_name']}</div>
                                      <div style="font-size:24px; font-weight:800; margin:6px 0;">{format_cost(alt_item['total_cost'])}</div>
                                      <div style="background:{soft_bg}; padding:9px; border-radius:8px; font-size:12px; margin-bottom:8px;">
                                        <b>{alt_item['carton_id']}</b> &nbsp;|&nbsp; {alt_item.get('estimated_days', '')}<br/>
                                        Util: {alt_item['utilization_pct']*100:.1f}% &nbsp;|&nbsp; Risk: {alt_item['risk_probability']*100:.1f}% &nbsp;|&nbsp; CO2: {alt_item.get('co2_kg', 0):.2f} kg
                                      </div>
                                      <div style="font-size:11.5px; color:{muted_col}; line-height:1.6;">• {cb_html}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                with t_sop:
                    c_sop, c_ai = st.columns(2)
                    with c_sop:
                        st.markdown("#### Warehouse Operator SOP")
                        st.markdown(sh.get("packing_instructions", ""))
                    with c_ai:
                        st.markdown("#### Engineering & AI Rationale")
                        st.markdown(sh.get("genai_explanation", ""))
                        st.caption(sh.get("carton_reasoning", ""))
                        st.caption(sh.get("risk_reasoning", ""))


# ------------------------------------------------------------------
# WORKSPACE 3: Master SKU & Box Registry (Database CRUD)
# ------------------------------------------------------------------
elif st.session_state.active_workspace == "Master SKU & Box Registry":
    st.markdown(
        """
        <div class="corp-banner">
          <h2 style="margin:0;">Master SKU & Container Fleet Registry</h2>
          <p style="margin:4px 0 0 0; opacity:0.82;">Persistent relational database for standard corporate SKUs and custom warehouse packaging assets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_skus, tab_boxes = st.tabs(["Master SKU Catalog (DB)", "Container Fleet & Custom Boxes (DB)"])

    with tab_skus:
        f_search, f_cat = st.columns([2, 1])
        q_str = f_search.text_input("Search SKUs", placeholder="Code or product name...")
        q_cat = f_cat.selectbox("Filter Category", ["All"] + CATEGORY_OPTIONS)

        skus, err = api_get("/skus", params={"search": q_str, "category": q_cat})
        if err:
            st.error(err)
        elif skus:
            df_skus = pd.DataFrame(skus)[
                [
                    "sku_code",
                    "name",
                    "product_category",
                    "length_cm",
                    "width_cm",
                    "height_cm",
                    "weight_kg",
                    "fragility_score",
                    "declared_value_inr",
                    "is_liquid",
                    "orientation_sensitive",
                    "requires_cold_chain",
                ]
            ]
            st.dataframe(df_skus, use_container_width=True, hide_index=True)

        c_add, c_del = st.columns([2, 1])
        with c_add.expander("Register or Update Master SKU", expanded=True):
            r1, r2, r3 = st.columns(3)
            new_code = r1.text_input("SKU Code*", placeholder="SKU-CORP-901")
            new_name = r2.text_input("Product Name*", placeholder="Precision Optical Sensor")
            new_cat = r3.selectbox("Category*", CATEGORY_OPTIONS, key="reg_cat")

            s1, s2, s3, s4, s5 = st.columns(5)
            nl = s1.number_input("L (cm)", min_value=1.0, value=20.0, key="reg_l")
            nw = s2.number_input("W (cm)", min_value=1.0, value=15.0, key="reg_w")
            nh = s3.number_input("H (cm)", min_value=1.0, value=12.0, key="reg_h")
            nwt = s4.number_input("Wt (kg)", min_value=0.05, value=1.5, key="reg_wt")
            nval = s5.number_input("Value (Rs.)", min_value=0.0, value=2500.0, key="reg_val")

            chk1, chk2, chk3 = st.columns(3)
            nliq = chk1.checkbox("Contains Liquid", key="reg_liq")
            nori = chk2.checkbox("Orientation Sensitive", key="reg_ori")
            ncold = chk3.checkbox("Requires Cold-Chain", key="reg_cold")

            if st.button("Save Master SKU to Database", type="primary"):
                if not new_code.strip() or not new_name.strip():
                    st.error("Both SKU Code and Name are required.")
                else:
                    _, perr = api_post(
                        "/skus",
                        {
                            "sku_code": new_code.strip(),
                            "name": new_name.strip(),
                            "product_category": new_cat,
                            "length_cm": nl,
                            "width_cm": nw,
                            "height_cm": nh,
                            "weight_kg": nwt,
                            "declared_value_inr": nval if nval > 0 else None,
                            "is_liquid": nliq,
                            "orientation_sensitive": nori,
                            "requires_cold_chain": ncold,
                        },
                    )
                    if perr:
                        st.error(perr)
                    else:
                        st.success(f"Saved {new_code.upper()} to database.")
                        st.rerun()

        with c_del.expander("Delete Master SKU", expanded=True):
            if skus:
                del_code = st.selectbox("Select SKU to Remove", [s["sku_code"] for s in skus])
                if st.button("Delete SKU", use_container_width=True):
                    _, derr = api_delete(f"/skus/{del_code}")
                    if derr:
                        st.error(derr)
                    else:
                        st.success(f"Deleted {del_code}.")
                        st.rerun()

    with tab_boxes:
        cat_data, cat_err = api_get("/catalog")
        if cat_err:
            st.error(cat_err)
        elif cat_data:
            containers = cat_data.get("containers", [])
            df_cont = pd.DataFrame(
                [
                    {
                        "ID": c["container_id"],
                        "Name": c["name"],
                        "Tier": c["tier_class"],
                        "Inner (cm)": f"{c['inner_dims_cm'][0]}x{c['inner_dims_cm'][1]}x{c['inner_dims_cm'][2]}",
                        "Max Payload (kg)": c["max_payload_kg"],
                        "Tare (kg)": c["tare_weight_kg"],
                        "Ply / Spec": f"{c['ply_rating']} | ECT {c['ect_kn_m']}",
                        "Cost (Rs.)": c["material_cost_inr"],
                        "Source": "Custom (DB)" if c.get("is_custom") else "BIS IS 2771",
                    }
                    for c in containers
                ]
            )
            st.dataframe(df_cont, use_container_width=True, hide_index=True)

        cb_add, cb_del = st.columns([2, 1])
        with cb_add.expander("Register Custom Warehouse Container", expanded=True):
            b1, b2, b3 = st.columns(3)
            cid = b1.text_input("Container ID*", placeholder="WH-CUSTOM-45")
            cname = b2.text_input("Container Name*", placeholder="Warehouse Special 45cm Box")
            ctier = b3.selectbox("Tier", ["parcel", "cold_parcel", "ptl"])

            d1, d2, d3, d4 = st.columns(4)
            il = d1.number_input("Inner L (cm)", min_value=5.0, value=45.0)
            iw = d2.number_input("Inner W (cm)", min_value=5.0, value=35.0)
            ih = d3.number_input("Inner H (cm)", min_value=5.0, value=25.0)
            maxp = d4.number_input("Max Payload (kg)", min_value=1.0, value=25.0)

            p1, p2, p3, p4 = st.columns(4)
            tare = p1.number_input("Tare Wt (kg)", min_value=0.0, value=0.9)
            ply = p2.text_input("Ply Rating", value="5-Ply")
            ect = p3.number_input("ECT (kN/m)", min_value=1.0, value=5.8)
            mcost = p4.number_input("Material Cost (Rs.)", min_value=1.0, value=115.0)

            if st.button("Register Custom Container", type="primary"):
                if not cid.strip() or not cname.strip():
                    st.error("Container ID and Name are required.")
                else:
                    _, berr = api_post(
                        "/containers/custom",
                        {
                            "container_id": cid.strip(),
                            "name": cname.strip(),
                            "tier_class": ctier,
                            "inner_l_cm": il,
                            "inner_w_cm": iw,
                            "inner_h_cm": ih,
                            "tare_weight_kg": tare,
                            "max_payload_kg": maxp,
                            "ply_rating": ply,
                            "ect_kn_m": ect,
                            "material_cost_inr": mcost,
                            "is_cold_chain": ctier == "cold_parcel",
                        },
                    )
                    if berr:
                        st.error(berr)
                    else:
                        st.success(f"Registered custom container {cid.upper()}.")
                        st.rerun()

        with cb_del.expander("Remove Custom Container", expanded=True):
            custom_list, _ = api_get("/containers/custom")
            if custom_list:
                del_cid = st.selectbox("Select Custom Box", [c["container_id"] for c in custom_list])
                if st.button("Delete Custom Box", use_container_width=True):
                    _, cerr = api_delete(f"/containers/custom/{del_cid}")
                    if cerr:
                        st.error(cerr)
                    else:
                        st.success(f"Deleted {del_cid}.")
                        st.rerun()
            else:
                st.caption("No custom warehouse boxes registered yet.")


# ------------------------------------------------------------------
# WORKSPACE 4: Audit Log & QA Telemetry
# ------------------------------------------------------------------
elif st.session_state.active_workspace == "Audit Log & QA Telemetry":
    st.markdown(
        """
        <div class="corp-banner">
          <h2 style="margin:0;">Dispatch Audit Ledger & Closed-Loop QA Telemetry</h2>
          <p style="margin:4px 0 0 0; opacity:0.82;">Review persisted dispatch optimizations, export printable warehouse manifests, and record real delivery outcomes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    orders, oerr = api_get("/orders", params={"limit": 100})
    if oerr:
        st.error(oerr)
    elif not orders:
        st.info("No dispatch orders recorded yet.")
    else:
        df_orders = pd.DataFrame(orders)[
            [
                "order_ref",
                "customer_or_po",
                "source_pin",
                "destination_pin",
                "shipping_zone_tier",
                "optimization_priority",
                "total_direct_cost",
                "cost_savings_inr",
                "total_co2_kg",
                "qa_status_summary",
            ]
        ]
        st.dataframe(df_orders, use_container_width=True, hide_index=True)

        st.divider()
        sel_ref = st.selectbox("Inspect Order & Record QA Outcome", [o["order_ref"] for o in orders])
        detail, derr = api_get(f"/orders/{sel_ref}")
        manifest, _ = api_get(f"/orders/{sel_ref}/manifest")

        if derr:
            st.error(derr)
        elif detail:
            top1, top2, top3 = st.columns([2, 1, 1])
            top1.markdown(
                f"**PO / Customer:** {detail['customer_or_po']} &nbsp;|&nbsp; "
                f"**Lane:** {detail['source_pin']} ({detail['source_zone']}) &rarr; "
                f"{detail['destination_pin']} ({detail['destination_zone']})"
            )
            if manifest:
                top2.download_button(
                    "Download HTML Manifest",
                    data=manifest["html_manifest"],
                    file_name=f"{sel_ref}_manifest.html",
                    mime="text/html",
                    use_container_width=True,
                )
                top3.download_button(
                    "Download Markdown SOP",
                    data=manifest["markdown_manifest"],
                    file_name=f"{sel_ref}_manifest.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            for sh in detail["shipments"]:
                with st.container(border=True):
                    s1, s2 = st.columns([1.4, 1])
                    with s1:
                        st.markdown(
                            f"#### Shipment #{sh['shipment_index'] + 1} — `{sh['recommended_carton_id']}` ({sh['group_label']})"
                        )
                        render_badge_strip(sh.get("handling_badges", []), is_dark)
                        st.caption(
                            f"Mode: {sh['transit_mode']} ({sh['estimated_transit_days']}) | "
                            f"Direct Cost: Rs.{sh['direct_cost']:,.0f} | "
                            f"Predicted Risk: {sh['damage_risk_probability']*100:.1f}% | "
                            f"CO2e: {sh['co2_emissions_kg']:.2f} kg"
                        )
                    with s2:
                        outcomes = ["pending", "delivered_intact", "damaged_in_transit"]
                        cur_idx = outcomes.index(sh["actual_outcome"]) if sh["actual_outcome"] in outcomes else 0
                        new_out = st.selectbox(
                            "Carrier QA Outcome",
                            outcomes,
                            index=cur_idx,
                            key=f"qa_sel_{sh['id']}",
                        )
                        new_notes = st.text_input(
                            "QA Notes",
                            value=sh.get("damage_notes", ""),
                            key=f"qa_note_{sh['id']}",
                        )
                        if st.button("Update QA Outcome", key=f"qa_btn_{sh['id']}", type="primary"):
                            _, uerr = api_patch(
                                f"/shipments/{sh['id']}/outcome",
                                {"actual_outcome": new_out, "damage_notes": new_notes},
                            )
                            if uerr:
                                st.error(uerr)
                            else:
                                st.success("QA outcome saved to database.")
                                st.rerun()


# ------------------------------------------------------------------
# WORKSPACE 5: Executive Analytics & ESG
# ------------------------------------------------------------------
elif st.session_state.active_workspace == "Executive Analytics & ESG":
    st.markdown(
        """
        <div class="corp-banner">
          <h2 style="margin:0;">Executive Analytics, Cost Savings & ESG Dashboard</h2>
          <p style="margin:4px 0 0 0; opacity:0.82;">Live aggregation of freight spend, optimization ROI, container fleet utilization, carbon reduction, and closed-loop ML calibration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics, aerr = api_get("/analytics/summary")
    if aerr:
        st.error(aerr)
    elif analytics:
        kpis = analytics["kpis"]
        qa = analytics["qa_calibration"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Audited Orders", f"{kpis['total_orders']}", delta=f"{kpis['total_shipments']} shipments", delta_color="off")
        m2.metric("Cumulative Freight Spend", format_cost(kpis["total_spend_inr"]))
        m3.metric("Realized Cost Savings", format_cost(kpis["total_savings_inr"]), delta=f"{kpis['avg_savings_pct']:.1f}% vs oversize")
        m4.metric("Carbon Footprint Saved", f"{kpis['total_co2_saved_kg']:.1f} kg CO2e", delta=f"{kpis['total_co2_kg']:.1f} kg emitted", delta_color="off")
        m5.metric(
            "QA Observed Damage",
            f"{qa['observed_damage_rate_pct']:.1f}%",
            delta=f"Predicted: {qa['mean_predicted_rate_pct']:.1f}% ({qa['evaluated_shipments']} QA)",
            delta_color="off",
        )

        st.divider()
        c_left, c_right = st.columns(2)

        trend_df = pd.DataFrame(analytics.get("recent_trend", []))
        with c_left:
            st.subheader("Spend vs Realized Optimization Savings")
            if not trend_df.empty:
                melted = trend_df.melt(
                    id_vars=["order_ref", "created_at"],
                    value_vars=["direct_cost", "savings_inr"],
                    var_name="Metric",
                    value_name="INR",
                )
                melted["Metric"] = melted["Metric"].map(
                    {"direct_cost": "Optimized Cost", "savings_inr": "Savings vs Oversize"}
                )
                chart1 = (
                    alt.Chart(melted)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("order_ref:N", sort=None, title="Order Ref"),
                        y=alt.Y("INR:Q", title="Amount (Rs.)"),
                        color=alt.Color(
                            "Metric:N",
                            scale=alt.Scale(
                                domain=["Optimized Cost", "Savings vs Oversize"],
                                range=["#38bdf8", "#10b981"],
                            ),
                        ),
                        tooltip=["order_ref", "created_at", "Metric", "INR"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart1, use_container_width=True)

        cont_df = pd.DataFrame(analytics.get("container_usage", []))
        with c_right:
            st.subheader("Container Fleet Mix & Avg Utilization")
            if not cont_df.empty:
                chart2 = (
                    alt.Chart(cont_df)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y("carton_id:N", sort="-x", title="Container"),
                        x=alt.X("shipments:Q", title="Shipment Count"),
                        color=alt.Color(
                            "avg_utilization_pct:Q",
                            title="Avg Util %",
                            scale=alt.Scale(scheme="tealblues"),
                        ),
                        tooltip=["carton_id", "shipments", "avg_utilization_pct", "total_cost"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart2, use_container_width=True)

        z_col, qa_col = st.columns([1.4, 1])
        zone_df = pd.DataFrame(analytics.get("zone_breakdown", []))
        with z_col:
            st.subheader("Zone A–E Cost & Transit Risk Profile")
            if not zone_df.empty:
                chart3 = (
                    alt.Chart(zone_df)
                    .mark_circle(size=220)
                    .encode(
                        x=alt.X("avg_cost:Q", title="Average Order Direct Cost (Rs.)"),
                        y=alt.Y("avg_risk_pct:Q", title="Average Predicted Risk (%)"),
                        size=alt.Size("orders:Q", title="Orders"),
                        color=alt.Color("zone_tier:N", title="Zone Tier"),
                        tooltip=["zone_tier", "orders", "avg_cost", "avg_risk_pct"],
                    )
                    .properties(height=270)
                )
                st.altair_chart(chart3, use_container_width=True)

        with qa_col:
            st.subheader("Closed-Loop QA Calibration & Fleet")
            st.markdown(
                f"""
                <div class="corp-card">
                  <div><b>Evaluated Carrier Shipments:</b> {qa['evaluated_shipments']}</div>
                  <div style="margin-top:6px;"><b>Delivered Intact:</b> {qa['intact_count']} &nbsp;|&nbsp; <b>Damaged:</b> {qa['damaged_count']}</div>
                  <div style="margin-top:6px;"><b>Observed Damage Rate:</b> {qa['observed_damage_rate_pct']:.2f}%</div>
                  <div style="margin-top:6px;"><b>Mean ML Predicted Risk:</b> {qa['mean_predicted_rate_pct']:.2f}%</div>
                  <hr style="border-color:{border_col}; margin:10px 0;"/>
                  <div><b>Master SKUs in DB:</b> {kpis['master_sku_count']}</div>
                  <div style="margin-top:4px;"><b>Custom Containers in DB:</b> {kpis['custom_container_count']}</div>
                  <div style="margin-top:4px;"><b>Fleet Mean Utilization:</b> {kpis['avg_net_utilization_pct']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
