import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Packaging Advisor API"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Illustrative prototype assumptions as of 2025-01-15, calibrated broadly
    # to private-courier per-kg pricing for small parcels in India. These are
    # not taken from any single published rate card.
    AIR_BASE_RATE_PER_KG: float = 60.0
    SURFACE_BASE_RATE_PER_KG: float = 15.0
    COURIER_RATE_CARD_DATE: str = "2025-01-15"

    class Config:
        case_sensitive = True


settings = Settings()
