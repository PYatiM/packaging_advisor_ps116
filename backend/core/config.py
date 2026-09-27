from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_HF_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 — 1.5B Ultra-Light Small",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5 — 3B Fast Small (Default)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 — 7B Instruct",
    "PYatiM/packaging_advisor_ps116": "Project Fine-Tuned Packaging Model",
}


class Settings(BaseSettings):
    """Application settings backed by environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Packaging Advisor Enterprise API"
    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "sqlite:///./data/packaging_advisor.db"

    GEMINI_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    HUGGINGFACE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"

    # Illustrative prototype assumptions as of 2025-01-15, calibrated broadly
    # to private-courier and B2B PTL per-kg pricing in India.
    AIR_BASE_RATE_PER_KG: float = 60.0
    SURFACE_BASE_RATE_PER_KG: float = 15.0
    PTL_BASE_RATE_PER_KG: float = 9.0
    PTL_CFT_KG_FACTOR: float = 7.0
    VOLUMETRIC_DIVISOR_PARCEL: float = 5000.0
    COURIER_RATE_CARD_DATE: str = "2025-01-15"


settings = Settings()
