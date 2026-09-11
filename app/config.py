"""Настройки приложения. Всё, что зависит от окружения, читается из переменных и .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "study_materials_v1"

    # Эмбеддинги
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    # Чанкинг (в токенах токенизатора эмбеддинг-модели)
    chunk_tokens: int = 500
    chunk_overlap: int = 60

    # Поиск
    top_k: int = 5
    # Порог косинусной близости: ниже — считаем, что в материалах ответа нет.
    # 0 — порог выключен, решает LLM. Подбирается на golden set.
    min_score: float = 0.0

    # LLM (OpenAI-совместимый API: Ollama, vLLM, внешний провайдер)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")
    llm_model: str = "qwen2.5:3b"
    llm_timeout: float = 60.0
    # С запасом: у рассуждающих моделей скрытые рассуждения тоже тратят этот лимит
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # Загрузка файлов
    data_dir: Path = Path("data")
    max_upload_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
