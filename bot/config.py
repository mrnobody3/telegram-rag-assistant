"""Модуль конфігурації застосунку на базі pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Налаштування оточення застосунку.

    Автоматично завантажує змінні з оточення або файлу .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Автентифікаційні ключі
    BOT_TOKEN: SecretStr = Field(
        ...,
        description="Токен Telegram-бота, отриманий від @BotFather",
    )
    GROQ_API_KEY: SecretStr = Field(
        ...,
        description="API ключ доступу до сервісу Groq",
    )

    # Шляхи до сховищ даних
    SQLITE_PATH: Path = Field(
        default=Path("data/db.sqlite3"),
        description="Шлях до файлу реляційної бази даних SQLite",
    )
    CHROMA_PATH: Path = Field(
        default=Path("data/chroma_data"),
        description="Шлях до директорії векторної бази даних ChromaDB",
    )

    # Моделі Groq
    GROQ_MODEL_LLM: str = Field(
        default="openai/gpt-oss-120b",
        description="Основна генеративна модель для відповідей на запитання",
    )
    GROQ_MODEL_EXTRACT: str = Field(
        default="openai/gpt-oss-20b",
        description="Швидка модель для екстракції та попередньої обробки даних",
    )

    @property
    def sqlite_url(self) -> str:
        """Повертає асинхронний DSN для SQLAlchemy (aiosqlite)."""
        return f"sqlite+aiosqlite:///{self.SQLITE_PATH.as_posix()}"

    def ensure_directories(self) -> None:
        """Створює необхідні директорії для зберігання даних, якщо вони відсутні."""
        self.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Повертає кешований екземпляр налаштувань застосунку."""
    settings = Settings()
    settings.ensure_directories()
    return settings


config: Settings = get_settings()
