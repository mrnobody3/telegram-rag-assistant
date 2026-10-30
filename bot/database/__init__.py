"""Модуль бази даних (моделі, сесії та репозиторії)."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.database.chroma_repo import VectorStore
    from bot.database.models import Base, Document
    from bot.database.sqlite_repo import DocumentRepository

__all__ = [
    "Base",
    "Document",
    "DocumentRepository",
    "VectorStore",
]


def __getattr__(name: str) -> Any:
    """Ледачий імпорт атрибутів пакета бази даних."""
    if name in ("Base", "Document"):
        from bot.database import models

        return getattr(models, name)
    if name == "DocumentRepository":
        from bot.database.sqlite_repo import DocumentRepository

        return DocumentRepository
    if name == "VectorStore":
        from bot.database.chroma_repo import VectorStore

        return VectorStore
    raise AttributeError(f"Модуль {__name__!r} не містить атрибута {name!r}")
