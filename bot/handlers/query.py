"""Обробник текстових запитань лісових звірів (aiogram 3.x)."""

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from bot.exceptions import AppException, RateLimitExceededError

if TYPE_CHECKING:
    from aiogram.types import Message
    from bot.database.models import Document
    from bot.database.sqlite_repo import DocumentRepository
    from bot.services.rag_engine import RAGService

try:
    from aiogram import F, Router
except ImportError:

    class _FallbackF:
        """Запасний об'єкт фільтра F для середовищ без встановленого aiogram."""

        def __getattr__(self, name: str) -> Any:
            return name

    class Router:  # type: ignore[no-redef]
        """Запасний клас Router для імпорту та юніт-тестування без встановленого aiogram."""

        def message(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

    F = _FallbackF()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = Router()


def format_answer_with_sources(
    answer: str,
    documents: Sequence[Any],
) -> str:
    """Чиста функція форматування відповіді LLM із додаванням списку джерел (автор, дата, файл)."""
    cleaned_answer = answer.strip()
    if not documents:
        return cleaned_answer

    source_lines: list[str] = []
    for doc in documents:
        file_name = getattr(doc, "file_name", "Невідомий файл")
        author_name = getattr(doc, "author_name", "Невідомий автор")
        created_at = getattr(doc, "created_at", None)
        if isinstance(created_at, datetime):
            date_str = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(created_at or "невідома дата")

        source_lines.append(f"• {file_name} — автор: {author_name} ({date_str})")

    sources_block = "\n".join(source_lines)
    return f"{cleaned_answer}\n\nДжерела:\n{sources_block}"


@router.message(F.text)
async def handle_text_query(
    message: "Message",
    rag_service: "RAGService",
    doc_repo: "DocumentRepository",
) -> None:
    """Обробляє звичайні текстові повідомлення як запитання до бази лісових знань.

    Викликає RAGService.answer_question, дістає метадані документів (автор, дата)
    з DocumentRepository за doc_id використаних чанків та надсилає відформатовану відповідь.
    """
    query = (message.text or "").strip()
    if not query:
        return

    try:
        answer = await rag_service.answer_question(query, location_filter=None)

        doc_ids: list[int] = list(getattr(rag_service, "last_used_doc_ids", []))
        documents: list["Document"] = []
        for doc_id in doc_ids:
            doc = await doc_repo.get_by_id(doc_id)
            if doc is not None:
                documents.append(doc)

        formatted_response = format_answer_with_sources(answer=answer, documents=documents)
        await message.answer(formatted_response)
    except RateLimitExceededError as exc:
        logger.warning("Перевищено ліміт запитів при обробці запитання '%s': %s", query, exc)
        await message.answer(exc.message)
    except AppException as exc:
        logger.warning("Помилка обробки запитання '%s': %s", query, exc)
        await message.answer(f"Не вдалося знайти відповідь: {exc.message}")
    except Exception as exc:
        logger.exception("Непередбачена помилка при обробці запитання '%s': %s", query, exc)
        await message.answer("Сталася внутрішня помилка під час пошуку відповіді.")
