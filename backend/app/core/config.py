import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env") if os.path.exists(os.path.join(PROJECT_ROOT, ".env")) else ".env"
CHROMA_DEFAULT_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "chroma_db")


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RRF_K: int = 60
    CHROMA_PERSIST_DIR: str = CHROMA_DEFAULT_DIR
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
