# Packaging Advisor PS116

This repository is an MVP for an AI-assisted packaging recommendation system. The prototype groups incompatible cargo, evaluates safe box sizes with an OR-Tools CP-SAT feasibility check, estimates damage risk from a synthetic packaging dataset, and proposes the lowest-cost shipping mode for each cart segment.

## Folder Structure

```
├── backend/
│   ├── api/          # FastAPI endpoints and recommendations logic
│   ├── core/         # app settings and environment-backed configuration
│   ├── genai/        # post-hoc explanation layer for recommendations
│   ├── ml/           # synthetic-data generator and trained damage-risk model
│   ├── optimization/ # OR-Tools CP-SAT 3D packing feasibility checks
│   ├── schemas/      # Pydantic models for carts and responses
│   ├── services/     # dynamic freight and packaging cost assumptions
│   └── main.py       # API bootstrap
├── deployment/       # Dockerfiles for api and UI services
├── frontend/         # Streamlit dashboard
├── results/          # evaluation output files and summary reports
├── scripts/          # synthetic data generation and ablation baselines
├── requirements.txt  # Python dependencies
├── docker-compose.yml
├── README.md
└── .gitignore
```

## MVP Description

The current implementation is a decision-support prototype, not a full logistics optimizer or production-grade freight engine. It does the following:

- groups SKU items by compatibility (perishable, liquids, electronics, and general cargo) to avoid unsafe mixing;
- evaluates each group against a small catalog of standard box and pallet sizes using an OR-Tools CP-SAT 3D placement model with axis-aligned rotations and pairwise non-overlap constraints;
- scores damage risk using a model trained on a synthetic dataset generated in the project (`backend/ml/dataset_generator.py`); the training pipeline compares RandomForest and XGBoost classifiers and keeps the best artifact in `backend/ml/artifacts/`;
- estimates shipping cost using illustrative assumed rates for this prototype as of 2025-01-15: air = Rs. 60/kg and surface = Rs. 15/kg. These are calibrated to be broadly consistent with private courier per-kg pricing for small parcels in India, but are not tied to any single published rate card. They can be overridden via `AIR_BASE_RATE_PER_KG` and `SURFACE_BASE_RATE_PER_KG` in the environment or settings object.

The project does not currently include a real production shipment API, a live tracking integration, or a full-blown mixed-integer optimizer for multi-warehouse planning.

## Rate Assumptions

The API defaults are illustrative assumed rates for this prototype as of 2025-01-15: air = Rs. 60/kg and surface = Rs. 15/kg. They are calibrated to be broadly consistent with private courier per-kg pricing for small parcels in India, but are not tied to any single published rate card. The values are represented as configuration settings so they can be reviewed and overridden without code edits.

## Running the Project Locally

Run the entire stack via Docker:

```bash
docker-compose up --build
```

Then visit:
- **Streamlit Dashboard:** http://localhost:8501
- **FastAPI Swagger:** http://localhost:8000/docs

## Evaluation

See the baseline comparison in [results/baseline_comparison.csv](results/baseline_comparison.csv) and the ablation report in [results/ablation_summary.md](results/ablation_summary.md). Across the synthetic test set, compatibility grouping materially lowered modeled freight cost and damage exposure relative to a no-grouping baseline, while the post-hoc GenAI explanation layer produced no numerical change to the underlying shipping or packing decisions—only the narrative explanation itself. The results support the current MVP conclusion that grouping and container fit are the dominant levers for cost and risk reduction, while LLM explanations are explanatory rather than decision-driving.