# Packaging Advisor PS116

This is the fully scaffolded MVP repository for the Enterprise AI Packaging Advisor.

## Folder Structure

```
├── backend/
│   ├── api/          # FastAPI Routes
│   ├── core/         # Settings & Config
│   ├── genai/        # OpenAI Explanations
│   ├── ml/           # XGBoost Predictors
│   ├── optimization/ # OR-Tools MIP Solvers
│   ├── schemas/      # Pydantic Data Models
│   └── services/     # Cost Engine
├── deployment/       # Dockerfiles
├── frontend/         # Streamlit UI
├── requirements.txt  # Project Dependencies
└── docker-compose.yml
```

## Running the Project Locally

Run the entire stack via Docker:

```bash
docker-compose up --build
```

Then visit:
- **Streamlit Dashboard:** http://localhost:8501
- **FastAPI Swagger:** http://localhost:8000/docs


