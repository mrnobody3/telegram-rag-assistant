"""Обробники завантаження документів та визначення локації (aiogram 3.x)."""

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot.exceptions import AppException, RateLimitExceededError
from bot.services.parser import ParserFactory
from bot.states.upload_states import UploadState

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from aiogram.types import Message
    from bot.database.sqlite_repo import DocumentRepository
    from bot.services.llm_groq import GroqService
    from bot.services.rag_engine import RAGService

try:
    from aiogram import F, Router
except ImportError:

    class _FallbackF:
        """Запасний об'єкт фільтра F для імпорту без встановленого aiogram."""

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

MISSING_LOCATION_PROMPT = (
    "Я не зрозумів, для якого місця ця порада. Напиши місце текстом"
)


def _extract_extension(file_name: str) -> str:
    """Чиста функція отримання розширення файлу з його назви."""
    return Path(file_name).suffix.lstrip(".").lower()


def _extract_author_info(message: Any) -> tuple[str, str]:
    """Чиста функція отримання ідентифікатора та імені автора з об'єкта Message."""
    user = getattr(message, "from_user", None)
    if user is None:
        return "unknown", "Невідомий звір"

    author_id = str(getattr(user, "id", "unknown"))
    author_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or "Лісовий мешканець"
    )
    return author_id, str(author_name)


async def _persist_and_index_document(
    *,
    file_name: str,
    location: str,
    author_id: str,
    author_name: str,
    text: str,
    doc_repo: "DocumentRepository",
    rag_service: "RAGService",
) -> int:
    """Єдина точка збереження документа в SQLite та індексації чанків у VectorStore.

    :return: ID створеного документа в БД.
    """
    await doc_repo.deactivate_old_versions(file_name=file_name, location=location)
    doc = await doc_repo.add_document(
        file_name=file_name,
        location=location,
        author_id=author_id,
        author_name=author_name,
    )
    await rag_service.process_and_save_document(
        text=text,
        doc_id=doc.id,
        location=location,
        extra_metadata={
            "file_name": file_name,
            "author_name": author_name,
        },
    )
    return int(doc.id)


@router.message(F.document)
async def handle_document_upload(
    message: "Message",
    state: "FSMContext",
    doc_repo: "DocumentRepository",
    groq_service: "GroqService",
    rag_service: "RAGService",
) -> None:
    """Обробляє повідомлення з прикріпленим документом.

    Завантажує файл у пам'ять, вилучає текст через ParserFactory та визначає локацію через GroqService.
    Якщо локацію не знайдено — переводить користувача у стан UploadState.waiting_for_location.
    """
    document = message.document
    if document is None or message.bot is None:
        return

    file_name = document.file_name or f"document_{document.file_id}.txt"
    extension = _extract_extension(file_name)
    author_id, author_name = _extract_author_info(message)

    try:
        parser = ParserFactory.get_parser(extension)

        buffer = io.BytesIO()
        await message.bot.download(document, destination=buffer)
        file_bytes = buffer.getvalue()

        text = parser.extract_text(file_bytes)
        if not text.strip():
            await message.answer("Не вдалося знайти текст у завантаженому файлі.")
            return

        location = await groq_service.extract_location(text[:2000])
        if location is None:
            await state.set_state(UploadState.waiting_for_location)
            await state.update_data(
                text=text,
                file_name=file_name,
                author_id=author_id,
                author_name=author_name,
            )
            await message.answer(MISSING_LOCATION_PROMPT)
            return

        doc_id = await _persist_and_index_document(
            file_name=file_name,
            location=location,
            author_id=author_id,
            author_name=author_name,
            text=text,
            doc_repo=doc_repo,
            rag_service=rag_service,
        )
        await state.clear()
        await message.answer(
            f"Документ '{file_name}' успішно збережено для локації '{location}' (ID: {doc_id})."
        )
    except RateLimitExceededError as exc:
        logger.warning("Перевищено ліміт запитів при завантаженні '%s': %s", file_name, exc)
        await message.answer(exc.message)
    except AppException as exc:
        logger.warning("Помилка обробки завантаженого документа '%s': %s", file_name, exc)
        await message.answer(f"Помилка обробки файлу: {exc.message}")
    except Exception as exc:
        logger.exception("Непередбачена помилка при завантаженні документа '%s': %s", file_name, exc)
        await message.answer("Сталася внутрішня помилка під час збереження документа.")


@router.message(UploadState.waiting_for_location, F.text)
async def handle_location_input(
    message: "Message",
    state: "FSMContext",
    doc_repo: "DocumentRepository",
    rag_service: "RAGService",
) -> None:
    """Приймає текстову назву локації від користувача у стані waiting_for_location та завершує індексацію."""
    location = (message.text or "").strip()
    if not location:
        await message.answer(MISSING_LOCATION_PROMPT)
        return

    data = await state.get_data()
    text = str(data.get("text") or "")
    file_name = str(data.get("file_name") or "unknown.txt")
    fallback_id, fallback_name = _extract_author_info(message)
    author_id = str(data.get("author_id") or fallback_id)
    author_name = str(data.get("author_name") or fallback_name)

    if not text:
        await state.clear()
        await message.answer("Дані документа втрачено. Будь ласка, надішліть файл ще раз.")
        return

    try:
        doc_id = await _persist_and_index_document(
            file_name=file_name,
            location=location,
            author_id=author_id,
            author_name=author_name,
            text=text,
            doc_repo=doc_repo,
            rag_service=rag_service,
        )
        await state.clear()
        await message.answer(
            f"Дякую! Документ '{file_name}' збережено для локації '{location}' (ID: {doc_id})."
        )
    except RateLimitExceededError as exc:
        logger.warning("Перевищено ліміт запитів у стані waiting_for_location: %s", exc)
        await message.answer(exc.message)
    except AppException as exc:
        logger.warning("Помилка збереження документа після введення локації: %s", exc)
        await message.answer(f"Помилка збереження документа: {exc.message}")
    except Exception as exc:
        logger.exception("Непередбачена помилка у стані waiting_for_location: %s", exc)
        await message.answer("Сталася внутрішня помилка під час збереження документа.")
