"""Настройки приложения. Всё, что зависит от окружения, читается из переменных и .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта: .env и data/ ищем от него, а не от текущей папки — API, интерфейс и CLI
# можно запускать откуда угодно. В Docker это /srv.
ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant. v2 — коллекция с dense и sparse векторами и структурным чанкингом
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "study_materials_v2"

    # Эмбеддинги
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    # Чанкинг (в токенах токенизатора эмбеддинг-модели)
    chunking: str = "structured"  # structured | simple
    chunk_tokens: int = 500
    chunk_overlap: int = 60
    chunk_min_tokens: int = 60

    # Поиск
    retrieval_mode: str = "hybrid"  # hybrid | dense
    sparse_language: str = "russian"
    top_k: int = 5
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # 10 на CPU: замер на лекциях — первое место то же, что при 20, а время 4,7 с вместо 8,6 с
    rerank_candidates: int = 10
    rerank_max_length: int = 512
    # Пороги по скору реранкера (сигмоида, 0..1): выше relevance_threshold — ответ в материалах есть,
    # ниже relevance_floor — точно нет, между ними спрашиваем LLM. Без реранкера порог не применяется.
    relevance_threshold: float = 0.3
    relevance_floor: float = 0.05
    # При отказе показываем «ближайшие темы» только с хоть какой-то релевантностью
    nearest_min_relevance: float = 0.01
    # Порог по косинусу для режима без реранкера. 0 — выключен, решает LLM.
    min_score: float = 0.0

    # Граф
    max_rewrites: int = 2
    recursion_limit: int = 30
    summary_trigger_tokens: int = 3000  # суммаризируем историю, когда она длиннее
    summary_keep_messages: int = 4  # столько последних сообщений остаются как есть
    quiz_questions: int = 5
    node_timeout: float = 120.0

    # LLM (OpenAI-совместимый API: Ollama, vLLM, внешний провайдер)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")
    llm_model: str = "qwen2.5:3b"
    llm_provider_name: str = "primary"
    # Запасные модели у того же провайдера, через запятую: free/gpt-6-luna,free/deepseek-v4.1-flash
    llm_fallback_models: str = ""
    # Запасной провайдер целиком (например, локальная Ollama), если основной лёг
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: SecretStr = SecretStr("not-needed")
    llm_fallback_model: str = ""
    llm_timeout: float = 60.0
    # С запасом: у рассуждающих моделей скрытые рассуждения тоже тратят этот лимит
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1
    llm_max_retries: int = 2
    # 429 с Retry-After дольше этого — не ждём, сразу идём к следующему провайдеру
    llm_max_retry_wait: float = 8.0

    # Данные
    data_dir: Path = ROOT / "data"
    max_upload_mb: int = 50

    # API и интерфейс
    api_url: str = "http://localhost:8000"  # куда ходит Streamlit
    public_api_url: str = "http://localhost:8000"  # адрес API для ссылок в браузере пользователя
    cors_origins: str = "http://localhost:8501"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def checkpoints_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def feedback_path(self) -> Path:
        return self.data_dir / "feedback.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
