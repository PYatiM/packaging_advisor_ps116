import streamlit as st
import requests
import os
import pandas as pd
import math
import json

# Default to the docker-compose service name 'api', allowing override via env var
API_URL = os.getenv("API_URL", "http://api:8000/api/v1")
CATEGORY_OPTIONS = [
    "Textiles & Apparel",
    "Precision Electronics",
    "Fine Art & Antiques",
    "Automotive Parts",
    "Perishable Goods",
    "Books & Media",
    "Industrial Machinery",
    "Glass & Ceramics",
]


def calculate_fragility(category, length, width, height, weight):
    """Heuristic fragility score based on category, density, and aspect ratio."""
    base_frag = 0.5
    if category in ["Fine Art & Antiques", "Glass & Ceramics"]:
        base_frag = 0.85
    elif category == "Precision Electronics":
        base_frag = 0.75
    elif category == "Perishable Goods":
        base_frag = 0.60
    elif category in ["Textiles & Apparel", "Books & Media"]:
        base_frag = 0.15
    elif category == "Automotive Parts":
        base_frag = 0.40
    elif category == "Industrial Machinery":
        base_frag = 0.25

    vol = length * width * height
    density = weight / (vol / 1000) if vol > 0 else 0

    if density > 0:
        density_factor = math.log10(density + 1) * 0.05
        base_frag -= density_factor

    dimensions = sorted([length, width, height])
    if dimensions[0] > 0:
        ratio = dimensions[2] / dimensions[0]
        ratio_penalty = (math.log(ratio) * 0.05) if ratio > 1 else 0
        base_frag += ratio_penalty

    return max(0.0, min(1.0, round(base_frag, 2)))


def format_volume(vol_cm3: float) -> str:
    """Display volume in m³ if large, otherwise cm³."""
    if vol_cm3 >= 1_000_000:
        return f"{vol_cm3 / 1_000_000:.2f} m\u00b3"
    return f"{vol_cm3:,.0f} cm\u00b3"


def format_cost(cost: float) -> str:
    """Format cost in lakhs if large, otherwise plain."""
    if cost >= 100_000:
        return f"Rs.{cost / 100_000:.2f}L"
    return f"Rs.{cost:,.0f}"


# ---------------------------------------------------------------
# Page config & session state
# ---------------------------------------------------------------
st.set_page_config(page_title="Packaging Optimizer", page_icon="📦", layout="wide")

if "cart" not in st.session_state:
    st.session_state.cart = []
if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#38bdf8"
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "model_provider" not in st.session_state:
    st.session_state.model_provider = "Gemini"
if "use_genai" not in st.session_state:
    st.session_state.use_genai = True

theme = st.session_state.theme_mode
page_bg = "#fff8ee" if theme == "Light" else "#101827"
panel_bg = "#ffffff" if theme == "Light" else "#172235"
soft_bg = "#ffe9d6" if theme == "Light" else "#24324a"
text_color = "#172235" if theme == "Light" else "#f8fbff"
alt_border = "#f0b58b" if theme == "Light" else "#38506d"
alt_text_muted = "#536277" if theme == "Light" else "#b7c5d8"
accent_color = st.session_state.accent_color

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background: {page_bg}; }}
    [data-testid="stHeader"] {{ background: {page_bg}; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: {panel_bg}; border-color: {alt_border}; }}
    [data-testid="stMetric"] {{ background: {panel_bg}; border: 1px solid {alt_border}; padding: 12px; border-radius: 8px; }}
    [data-testid="stAlert"] {{ border-radius: 8px; }}
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown, [data-testid="stText"] {{ color: {text_color} !important; }}

    /* Metrics */
    [data-testid="stMetricValue"] {{ color: {accent_color} !important; font-weight: 800; }}

    /* Buttons */
    .stButton button {{ transition: all 0.3s ease !important; border-radius: 8px !important; }}
    .stButton button[kind="primary"] {{ background: {accent_color} !important; border: 2px solid {accent_color} !important; color: #101827 !important; font-weight: 800 !important; }}
    .stButton button[kind="primary"]:hover {{ background: #ff6b5e !important; border-color: #ff6b5e !important; color: white !important; }}

    /* Tabs */
    [data-baseweb="tab"][aria-selected="true"] {{ border-bottom-color: {accent_color} !important; }}
    [data-baseweb="tab"][aria-selected="true"] p {{ color: {accent_color} !important; font-weight: 800 !important; }}

    /* Inputs & Interactivity */
    .stTextInput input:focus, .stNumberInput input:focus, [data-baseweb="select"]:focus-within {{ border-color: {accent_color} !important; }}
    [data-testid="stExpander"] summary:hover p {{ color: {accent_color} !important; }}
    hr {{ border-color: {accent_color} !important; }}

    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("AI Smart Packaging Optimizer")
st.markdown(
    "Optimal box selection, shipping cost estimation, and damage-risk analysis "
    "using 3D bin packing and Generative AI."
)
st.divider()

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_cart, tab_insights, tab_settings = st.tabs([
    "Cart Management", "Packaging Insights", "Settings"
])

# ===================== TAB 1: CART ==============================
with tab_cart:
    cart_count = len(st.session_state.cart)
    cart_weight = sum(item["weight_kg"] for item in st.session_state.cart)
    cart_volume = sum(
        item["length_cm"] * item["width_cm"] * item["height_cm"]
        for item in st.session_state.cart
    )
    summary_1, summary_2, summary_3 = st.columns(3)
    summary_1.metric("Items", cart_count)
    summary_2.metric("Total weight", f"{cart_weight:,.1f} kg")
    summary_3.metric("Total volume", format_volume(cart_volume))
    st.markdown("### Build your order")
    st.caption("Add products on the left, then review and edit the order on the right.")
    col_form, col_table = st.columns([1, 2.5])

    with col_form:
        with st.container():
            st.subheader("Add item")
            sku = st.text_input("SKU ID", placeholder="SKU-1001")
            category = st.selectbox("Category", CATEGORY_OPTIONS)

            c1, c2 = st.columns(2)
            length = c1.number_input("L (cm)", min_value=1.0, value=10.0, step=1.0)
            width = c2.number_input("W (cm)", min_value=1.0, value=10.0, step=1.0)
            height = c1.number_input("H (cm)", min_value=1.0, value=10.0, step=1.0)
            weight = c2.number_input("Wt (kg)", min_value=0.1, value=1.0, step=0.1)

            is_liquid = st.checkbox("Contains Liquid")
            orientation_sensitive = st.checkbox("Orientation Sensitive (This Way Up)")

            if st.button("Add to Cart", use_container_width=True, type="primary"):
                if not sku.strip():
                    st.error("SKU ID required!")
                else:
                    frag = calculate_fragility(category, length, width, height, weight)
                    st.session_state.cart.append({
                        "sku_id": sku,
                        "product_category": category,
                        "length_cm": length,
                        "width_cm": width,
                        "height_cm": height,
                        "weight_kg": weight,
                        "fragility_score": frag,
                        "is_liquid": is_liquid,
                        "orientation_sensitive": orientation_sensitive,
                    })
                    st.experimental_rerun()

        with st.expander("Bulk JSON load"):
            json_dump = st.text_area("Paste JSON Array", height=100)
            if st.button("Load JSON", use_container_width=True):
                try:
                    items = json.loads(json_dump)
                    if not isinstance(items, list):
                        raise ValueError("JSON must contain an array of items.")
                    for item in items:
                        if not isinstance(item, dict):
                            raise ValueError("Each array entry must be an object.")
                        if "fragility_score" not in item:
                            item["fragility_score"] = calculate_fragility(
                                item.get("product_category", "Books & Media"),
                                item.get("length_cm", 10),
                                item.get("width_cm", 10),
                                item.get("height_cm", 10),
                                item.get("weight_kg", 1),
                            )
                        st.session_state.cart.append(item)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    with col_table:
        st.subheader("Review order")
        if st.session_state.cart:
            df = pd.DataFrame(st.session_state.cart)
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                disabled=["fragility_score"],
            )
            st.session_state.cart = edited_df.to_dict("records")

            total_wt = sum(item["weight_kg"] for item in st.session_state.cart)
            total_vol = sum(
                item["length_cm"] * item["width_cm"] * item["height_cm"]
                for item in st.session_state.cart
            )
            st.caption(f"{len(st.session_state.cart)} items | {total_wt:,.1f} kg | {format_volume(total_vol)}")

            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("Remove Last Item", use_container_width=True) and st.session_state.cart:
                st.session_state.cart.pop()
                st.experimental_rerun()
            if c_btn2.button("Clear Cart", use_container_width=True):
                st.session_state.cart = []
                st.experimental_rerun()
        else:
            st.info("Cart is empty. Add items manually or load a JSON array.")

# ===================== TAB 2: INSIGHTS ==========================
with tab_insights:
    st.subheader("Optimise & analyse")
    st.caption("Set the route, then generate a packing and freight recommendation.")

    with st.container():
        col_p1, col_p2 = st.columns(2)
        source_pin = col_p1.text_input(
        "Source PIN Code", value="110001", help="6-digit Indian PIN code"
        )
        dest_pin = col_p2.text_input(
        "Destination PIN Code", value="560001", help="6-digit Indian PIN code"
        )

    if st.button("Run Optimisation", type="primary", use_container_width=True):
        if not st.session_state.cart:
            st.warning("Add items to cart first.")
        elif not (source_pin.isdigit() and len(source_pin) == 6 and dest_pin.isdigit() and len(dest_pin) == 6):
            st.error("Enter valid six-digit source and destination PIN codes.")
        else:
            with st.spinner("Running 3D Bin Packing & cost analysis..."):
                payload = {
                    "items": st.session_state.cart,
                    "source_pin": source_pin,
                    "destination_pin": dest_pin,
                    "use_genai": st.session_state.use_genai,
                    "model_provider": st.session_state.model_provider,
                }
                try:
                    res = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()

                        st.success("Optimisation complete.")

                        if data.get("compatibility_warning"):
                            st.warning(data["compatibility_warning"])

                        # Route & summary metrics
                        st.markdown("---")
                        route_col1, route_col2, metric_col = st.columns([1.5, 1.5, 1])

                        src_zone = data.get("source_zone", "Unknown")
                        dst_zone = data.get("destination_zone", "Unknown")

                        with route_col1:
                            st.markdown("### Origin")
                            st.markdown(f"**PIN:** {source_pin} | **Zone:** {src_zone}")

                        with route_col2:
                            st.markdown("### Destination")
                            st.markdown(f"**PIN:** {dest_pin} | **Zone:** {dst_zone}")

                        with metric_col:
                            st.metric(
                                "Total Cart Cost",
                                format_cost(data["total_cart_cost"]),
                                delta=f"{len(data['shipments'])} shipment(s)",
                                delta_color="off",
                            )

                        st.markdown("---")

                        # Shipment breakdown
                        st.subheader("Shipment Breakdown")
                        for idx, shipment in enumerate(data["shipments"]):
                            ship_cost = format_cost(shipment["total_estimated_cost"])
                            header = (
                                f"Shipment {idx+1} — "
                                f"{shipment['recommended_carton_id']} | "
                                f"{ship_cost}"
                            )
                            with st.expander(header, expanded=True):
                                st.info(
                                    f"**Transit Mode:** "
                                    f"{shipment.get('transit_mode_advice', 'Standard')}"
                                )
                                st.write("")

                                alts = shipment.get("alternatives", [])
                                if alts:
                                    st.markdown("#### Cost Alternatives")
                                    alt_cols = st.columns(len(alts))
                                    for a_idx, alt in enumerate(alts):
                                        with alt_cols[a_idx]:
                                            formatted_cb = alt['cost_breakdown'].replace(" | ", "<br>• ")
                                            st.markdown(f"""
<div style='border: 2px solid {alt_border}; border-radius: 8px; padding: 20px; background: {panel_bg}; height: 100%; margin-bottom: 15px;'>
    <h4 style='margin-top:0; color:{accent_color}; margin-bottom: 15px;'>{alt['tier_name']}</h4>
    <h2 style='margin:0; color:{text_color}; margin-bottom: 15px;'>{format_cost(alt['total_cost'])}</h2>

    <div style='background: {soft_bg}; padding: 12px; border-radius: 6px; margin-bottom: 15px; border: 1px solid {alt_border};'>
        <p style='color:{text_color}; font-size:0.95em; margin:0 0 5px 0;'>Container: <b>{alt['carton_id']}</b></p>
        <p style='margin:0; font-size:0.85em; color:{text_color}'>Utilisation: {alt['utilization_pct']*100:.1f}% &nbsp;|&nbsp; Risk: {alt['risk_probability']*100:.1f}%</p>
    </div>

    <p style='color:{alt_text_muted}; font-size:0.85em; line-height: 1.8; margin: 0;'>• {formatted_cb}</p>
</div>
                                            """, unsafe_allow_html=True)

                                st.write("")
                                st.markdown("#### Detailed Analysis")
                                it1, it2 = st.tabs(["AI Reasoning", "Packing Instructions"])
                                with it1:
                                    st.write("")
                                    st.markdown(shipment["genai_explanation"])
                                    st.divider()
                                    st.markdown(f"> **System Reasoning:** {shipment.get('carton_reasoning', '')}")
                                    st.markdown(f"> **Risk Factor:** {shipment.get('risk_reasoning', '')}")
                                    st.write("")
                                with it2:
                                    st.write("")
                                    st.markdown(
                                        shipment.get(
                                            "packing_instructions",
                                            "No specific instructions.",
                                        )
                                    )
                                    st.write("")

                    else:
                        st.error(f"Backend Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

# ===================== TAB 3: SETTINGS ==========================
with tab_settings:
    st.subheader("System Settings")

    st.markdown("### UI Theme")

    col_t1, col_t2 = st.columns(2)
    theme_choice = col_t1.radio(
        "Mode", 
        ["Dark", "Light"], 
        index=0 if st.session_state.theme_mode == "Dark" else 1, 
        horizontal=True
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.experimental_rerun()

    accent = col_t2.color_picker("Accent Colour", value=st.session_state.accent_color)
    if accent != st.session_state.accent_color:
        st.session_state.accent_color = accent
        st.experimental_rerun()

    st.markdown("### AI Preferences")
    st.session_state.model_provider = st.selectbox(
        "LLM Provider",
        ["Gemini", "Hugging Face"],
        index=0 if st.session_state.model_provider == "Gemini" else 1,
    )

    if st.session_state.model_provider == "Hugging Face":
        st.info(
            "Hugging Face selected. Ensure `HUGGINGFACE_API_KEY` is set in your `.env` file."
        )
    else:
        st.info("Gemini selected. Ensure `GEMINI_API_KEY` is set in your `.env` file.")

    with st.expander("Container & Transport Directory"):
        st.markdown("""
| ID | Dimensions (cm) | Max Weight | Material Cost | Type |
|---|---|---|---|---|
| C-SMALL-01 | 15 x 15 x 15 | 2 kg | Rs.40 | Carton |
| C-MED-05 | 30 x 30 x 20 | 5 kg | Rs.95 | Carton |
| C-LARGE-99 | 50 x 50 x 40 | 15 kg | Rs.210 | Carton |
| C-XL-100 | 100 x 100 x 100 | 50 kg | Rs.400 | Carton |
| PALLET-STD | 120 x 100 x 150 | 1,000 kg | Rs.1,500 | Pallet |
| PTL-REEFER | 590 x 235 x 239 | 5,000 kg | Rs.5,000 | Reefer |
| TRUCK-20FT | 590 x 235 x 239 | 15,000 kg | Rs.10,000 | Truck |
| TRUCK-40FT-HC | 1203 x 235 x 270 | 28,000 kg | Rs.15,000 | Truck |
        """)
