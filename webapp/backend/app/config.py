from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "CofPred API"
    api_prefix: str = "/api"
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'cofpred_web.db'}"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    master_data_path: Path = REPO_ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
    model_registry_path: Path = REPO_ROOT / "models" / "model_registry.csv"

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()

