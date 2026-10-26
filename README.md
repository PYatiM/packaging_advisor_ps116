# PackOptima Enterprise — Packaging Advisor (PS116 v2.5)

**PackOptima Enterprise** is a standards-grounded packaging optimization, Indian courier/freight rating, and transit-damage prevention platform. It combines **cargo compatibility segregation**, **OR-Tools CP-SAT 3D bin packing with ISTA 3A cushioning**, a **BIS IS 2771 (Part 1): 2022 + cold-chain container catalog**, **Indian 6-digit PIN Zone A–E routing**, **Scope 3 carbon accounting**, a **calibrated 15-feature ML damage predictor**, **SQLAlchemy relational persistence**, and a **post-hoc multi-LLM explanation layer** supporting both **Google Gemini** and **Hugging Face small Qwen models**.

---

## Key Capabilities

1. **Smart Cargo Compatibility Grouping ([grouping.py](backend/services/grouping.py)):**
   Isolates incompatible cargo into dedicated shipments (*perishables*, *liquids*, *electronics*, *fragile_delicate*, *heavy_industrial*, and consolidated *general* dry cargo) to eliminate liquid leaks, thermal spoilage, ESD shock, and heavy-vs-fragile impact crushing.
2. **Standards-Grounded 3D CP-SAT Packing & Multi-Box Splitting ([solver.py](backend/optimization/solver.py)):**
   - Expands `SKUItem.quantity` and adds **ISTA 3A per-face protective clearance** (`0.0–3.0 cm` based on fragility and mass).
   - Verifies exact 3D placement with 6 axis-aligned rotations (`orientation_sensitive` locks height along the vertical `z`-axis), pairwise non-overlap constraints, and gravity-weighted corner packing (falling back to 3D shelf-FFD above 35 units).
   - Supports **multi-carton splitting (`2–5` cartons)** and dynamic **custom warehouse containers** registered in the database.
3. **Calibrated Indian Logistics Cost, Cold-Chain & Scope 3 Carbon Model ([cost_engine.py](backend/services/cost_engine.py), [routing_engine.py](backend/services/routing_engine.py)):**
   - Maps 6-digit Indian PIN codes into **Zones A–E** (Local NCR/MMR/Metro Clusters, Regional, Metro-to-Metro, National Cross-Country, and Special/ODA regions including North-East `78–79`, J&K `18–19`, HP `17`, islands, and APS `90–99`).
   - Applies parcel `L × W × H / 5000` volumetric divisors, slab rounding, `10 kg+` weight tapering, realistic void-fill dunnage costs, stacked-height B2B pallet `7 kg/CFT` density billing, and realistic FTL lane multipliers.
   - Separates **passive EPS + PCM cold-chain parcels (`CC-EPS-S/M/L`)** for small/medium perishables from **active `PTL-REEFER` (>35 kg)**, restricting passive gel-pack parcels from multi-day ambient Zone C–E surface legs.
   - Tracks Scope 3 CO2e emissions and compares against a realistic `+2` container-size unoptimized baseline.
4. **Calibrated 15-Feature ML Transit Damage Predictor ([train_damage_model.py](backend/ml/train_damage_model.py), [predictor.py](backend/ml/predictor.py)):**
   - Trains and compares **XGBoost**, **RandomForest**, and **HistGradientBoosting** with 3-fold **isotonic regression (`CalibratedClassifierCV`)** across 12,000 physics-grounded shipments (19.71% damaged prevalence).
   - Best calibrated model (**XGBoost**): **ROC-AUC `0.8919`**, **Brier Score `0.0848`**, **Accuracy `88.96%`**, **Avg Precision `0.7684`**.
5. **Post-Hoc LLM Switching: Google Gemini & Hugging Face Small Qwen Models ([service.py](backend/genai/service.py)):**
   - Executes **strictly after** geometric packing, routing, and ML scoring complete (`0.00%` alteration to numbers or container choices).
   - Switch dynamically in the UI sidebar or `.env` between:
     - **Google Gemini (`google-genai`):** `gemini-2.5-flash` (default) or `gemini-2.5-pro`.
     - **Hugging Face Serverless (`InferenceClient`):** `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` (default small Qwen), `Qwen/Qwen2.5-7B-Instruct`, `PYatiM/packaging_advisor_ps116`, or any custom HF repository ID—with automatic deterministic engineering fallback when offline or unconfigured.
6. **Relational Database & 5-Workspace Control Tower UI ([models.py](backend/db/models.py), [app.py](frontend/app.py)):**
   - SQLAlchemy 2.0 ORM (`master_skus`, `custom_containers`, `order_audits`, `shipment_records`) seeded on boot with 12 enterprise SKUs and 12 historical audited orders.
   - 5 Streamlit workspaces: *Order Studio & Dispatch*, *3D Load Plan & Optimization* (interactive HTML5 3D viewer with mouse orbit, zoom, exploded view, and ISTA cushion shell toggle), *Master SKU & Box Registry*, *Audit Log & QA Telemetry* (printable HTML/Markdown manifests + closed-loop carrier QA outcomes), and *Executive Analytics & ESG* (Altair charts).

---

## Repository Structure

```text
├── backend/
│   ├── api/                 # FastAPI core (/recommend, /health, /catalog) & enterprise CRUD/analytics routes
│   ├── core/                # Environment settings & single-source BIS IS 2771 / ISTA 3A catalog
│   ├── db/                  # SQLAlchemy ORM models, SQLite/Postgres session & startup seeding
│   ├── genai/               # Post-hoc Gemini 2.5 & Hugging Face Qwen2.5 explanation service
│   ├── ml/                  # 15-feature synthetic generator, isotonic trainer & cached predictor
│   ├── optimization/        # OR-Tools CP-SAT 3D packer, FFD fallback & multi-carton splitter
│   ├── schemas/             # Pydantic v2 request/response contracts
│   └── services/            # Compatibility grouping, Zone A-E PIN router & freight/CO2e engine
├── deployment/              # Split API and slim UI Dockerfiles
├── docs/                    # IEEE research paper (PDF + LaTeX), 10-page report (PDF + MD) & figures
├── frontend/                # PackOptima Enterprise 5-workspace Streamlit Control Tower
├── results/                 # Empirical 200-cart baseline comparison & ablation reports
├── scripts/                 # Benchmarks, ablations, synthetic generator & report/paper builder
├── tests/                   # 36 automated pytest unit & end-to-end integration tests
├── .env.example             # Environment variable template (copy to .env)
├── docker-compose.yml       # Multi-container stack with persistent ./data volume
├── requirements.txt         # Full backend + ML + UI dependencies
├── requirements-ui.txt      # Slim UI-only dependencies for Dockerfile.ui
└── run_app.py               # 1-command local launcher (API + UI + --check health mode)
```

---

## Setup & Running Locally

### 1. Environment Configuration
Copy `.env.example` to `.env` and add optional LLM keys (the entire optimization, ML, and database stack runs deterministically even without keys):

```bash
cp .env.example .env
```

Key environment variables:
- `GEMINI_API_KEY` / `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `HUGGINGFACE_API_KEY` / `HUGGINGFACE_MODEL` (default: `Qwen/Qwen2.5-3B-Instruct`; also supports `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, or custom repos)
- `DATABASE_URL` (default: `sqlite:///./data/packaging_advisor.db`)
- `AIR_BASE_RATE_PER_KG` (`60.0`), `SURFACE_BASE_RATE_PER_KG` (`15.0`), `PTL_BASE_RATE_PER_KG` (`9.0`)

### 2. One-Command Turnkey Launch (Recommended)
```bash
pip install -r requirements.txt
python run_app.py
```
- **Control Tower UI:** http://localhost:8501
- **FastAPI Swagger Docs:** http://localhost:8000/docs
- Automated smoke check (boots both services, verifies HTTP 200 health endpoints, exits cleanly):
  ```bash
  python run_app.py --check
  ```

### 3. Docker Compose
```bash
docker-compose up --build
```

---

## Testing, Training & Benchmarks

```bash
# 1. Run 36 automated unit & integration tests
python -m pytest tests/ -v

# 2. Retrain & isotonically calibrate the 15-feature damage model
python -m backend.ml.train_damage_model

# 3. Run 200-cart empirical baseline comparison & ablation studies
python scripts/evaluate_baselines.py --ablations

# 4. Regenerate IEEE research paper, LaTeX source, and 10-page technical report
python scripts/generate_ieee_paper_and_report.py
```

### Empirical Benchmark Summary (200 Carts, Seed 42)
| Method | Avg Utilization | Avg Direct Cost | Avg Damage Risk | Wilcoxon p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed Pipeline (Ours)** | **28.6%** | **Rs. 1,138.63** | **0.112** | — |
| Always-Largest-Box (`C-XL-100`) | 1.2% | Rs. 11,264.18 | 0.200 | `p < 0.001` |
| Simulated Manual Baseline | 46.0% | Rs. 634.58 | 0.437 | `p < 0.001` |
| No-Grouping Baseline | 31.3% | Rs. 644.97 | 0.321 | `p < 0.001` |

---

## Research Paper & Project Documentation
- **IEEE Conference Paper (PDF):** [docs/IEEE_Research_Paper_PackOptima_PS116.pdf](docs/IEEE_Research_Paper_PackOptima_PS116.pdf)
- **IEEE Conference LaTeX Source:** [docs/IEEE_Research_Paper_PackOptima_PS116.tex](docs/IEEE_Research_Paper_PackOptima_PS116.tex)
- **10-Page Technical Project Report (PDF):** [docs/Project_Report_10Pages_PackOptima_PS116.pdf](docs/Project_Report_10Pages_PackOptima_PS116.pdf)
- **Ablation & Sensitivity Report:** [results/ablation_summary.md](results/ablation_summary.md)
