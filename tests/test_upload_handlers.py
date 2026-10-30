"""Тести для FSM станів та обробників завантаження документів (bot/handlers/upload.py)."""

import io
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from bot.handlers.upload import (
    handle_cancel_upload,
    handle_document_upload,
    handle_location_command_warning,
    handle_location_input,
)
from bot.states.upload_states import UploadState


class FakeFSMContext:
    """Тестовий аналог FSMContext для перевірки переходів станів та збережених даних."""

    def __init__(self) -> None:
        self.state: Any = None
        self.data: dict[str, Any] = {}

    async def set_state(self, state: Any) -> None:
        self.state = state

    async def update_data(self, **kwargs: Any) -> dict[str, Any]:
        self.data.update(kwargs)
        return dict(self.data)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def get_state(self) -> Any:
        return self.state

    async def clear(self) -> None:
        self.state = None
        self.data.clear()


def _make_document_message(
    file_name: str = "advice.txt",
    file_bytes: bytes = "Збирайте гриби у березовому гаю.".encode("utf-8"),
    file_size: int = 1024,
    user_id: int = 42,
    full_name: str = "Їжачок",
) -> MagicMock:
    """Створює мок об'єкта Message із прикріпленим документом."""
    message = MagicMock()
    message.document = MagicMock()
    message.document.file_id = "file_123"
    message.document.file_name = file_name
    message.document.file_size = file_size

    message.from_user = MagicMock()
    message.from_user.id = user_id
    message.from_user.full_name = full_name
    message.from_user.username = "hedgehog"

    async def _download(file: Any, destination: io.BytesIO | None = None) -> io.BytesIO:
        buf = destination if destination is not None else io.BytesIO()
        buf.write(file_bytes)
        buf.seek(0)
        return buf

    message.bot = MagicMock()
    message.bot.download = AsyncMock(side_effect=_download)
    message.answer = AsyncMock()
    return message


class TestUploadHandlers(unittest.IsolatedAsyncioTestCase):
    """Тести обробників завантаження документа та ручного введення локації."""

    def setUp(self) -> None:
        self.state = FakeFSMContext()

        self.doc_repo = MagicMock()
        self.doc_repo.deactivate_old_versions = AsyncMock(return_value=1)
        created_doc = MagicMock()
        created_doc.id = 101
        self.doc_repo.add_document = AsyncMock(return_value=created_doc)

        self.groq_service = MagicMock()
        self.groq_service.extract_location = AsyncMock()

        self.rag_service = MagicMock()
        self.rag_service.process_and_save_document = AsyncMock(return_value=["doc_101_chunk_0"])

    async def test_upload_rejects_file_larger_than_20mb(self) -> None:
        """Якщо розмір файлу перевищує 20 МБ, бот відповідає про обмеження і не завантажує файл."""
        too_large_size = 20 * 1024 * 1024 + 1
        message = _make_document_message(file_size=too_large_size)

        await handle_document_upload(
            message=message,
            state=self.state,
            doc_repo=self.doc_repo,
            groq_service=self.groq_service,
            rag_service=self.rag_service,
        )

        message.answer.assert_awaited_once_with(
            "Цей сувій занадто важкий для мене, приймаю до 20 МБ."
        )
        message.bot.download.assert_not_awaited()
        self.doc_repo.add_document.assert_not_awaited()

    async def test_upload_with_detected_location_saves_immediately(self) -> None:
        """Якщо GroqService визначив локацію, документ одразу зберігається у БД та VectorStore."""
        self.groq_service.extract_location.return_value = "березовий гай"
        message = _make_document_message()

        await handle_document_upload(
            message=message,
            state=self.state,
            doc_repo=self.doc_repo,
            groq_service=self.groq_service,
            rag_service=self.rag_service,
        )

        self.doc_repo.deactivate_old_versions.assert_awaited_once_with(
            file_name="advice.txt",
            location="березовий гай",
        )
        self.doc_repo.add_document.assert_awaited_once_with(
            file_name="advice.txt",
            location="березовий гай",
            author_id="42",
            author_name="Їжачок",
        )
        self.rag_service.process_and_save_document.assert_awaited_once()
        self.assertIsNone(self.state.state)

    async def test_upload_without_location_enters_waiting_state(self) -> None:
        """Якщо локація None, текст зберігається в FSM і бот просить вказати місце."""
        self.groq_service.extract_location.return_value = None
        message = _make_document_message()

        await handle_document_upload(
            message=message,
            state=self.state,
            doc_repo=self.doc_repo,
            groq_service=self.groq_service,
            rag_service=self.rag_service,
        )

        self.assertEqual(self.state.state, UploadState.waiting_for_location)
        self.assertEqual(self.state.data.get("file_name"), "advice.txt")
        self.assertIn("Збирайте гриби", self.state.data.get("text", ""))
        self.doc_repo.add_document.assert_not_awaited()

        message.answer.assert_awaited_with(
            "Я не зрозумів, для якого місця ця порада. Напиши місце текстом"
        )

    async def test_waiting_for_location_saves_document_and_clears_state(self) -> None:
        """У стані waiting_for_location введений текст стає локацією і документ зберігається."""
        await self.state.set_state(UploadState.waiting_for_location)
        await self.state.update_data(
            text="Ховайте горіхи у дуплі старого дуба.",
            file_name="squirrel.txt",
            author_id="77",
            author_name="Білочка",
        )

        location_msg = MagicMock()
        location_msg.text = "дубовий ліс"
        location_msg.from_user = MagicMock()
        location_msg.from_user.id = 77
        location_msg.from_user.full_name = "Білочка"
        location_msg.answer = AsyncMock()

        await handle_location_input(
            message=location_msg,
            state=self.state,
            doc_repo=self.doc_repo,
            rag_service=self.rag_service,
        )

        self.doc_repo.deactivate_old_versions.assert_awaited_once_with(
            file_name="squirrel.txt",
            location="дубовий ліс",
        )
        self.doc_repo.add_document.assert_awaited_once_with(
            file_name="squirrel.txt",
            location="дубовий ліс",
            author_id="77",
            author_name="Білочка",
        )
        self.rag_service.process_and_save_document.assert_awaited_once()
        self.assertIsNone(self.state.state)

    async def test_waiting_for_location_command_warns_and_does_not_save(self) -> None:
        """Команда зі слешем у стані waiting_for_location повертає попередження і НЕ зберігає документ."""
        await self.state.set_state(UploadState.waiting_for_location)
        await self.state.update_data(
            text="Ховайте горіхи у дуплі старого дуба.",
            file_name="squirrel.txt",
            author_id="77",
            author_name="Білочка",
        )

        cmd_msg = MagicMock()
        cmd_msg.text = "/help"
        cmd_msg.answer = AsyncMock()

        await handle_location_command_warning(message=cmd_msg)

        cmd_msg.answer.assert_awaited_once_with(
            "Будь ласка, введіть назву локації звичайним текстом (без '/'). "
            "Або натисніть /cancel для скасування процесу."
        )
        self.doc_repo.add_document.assert_not_awaited()
        self.assertEqual(self.state.state, UploadState.waiting_for_location)

    async def test_cancel_command_clears_state_and_notifies_user(self) -> None:
        """Команда /cancel очищує FSM-стан завантаження документа та повідомляє про скасування."""
        await self.state.set_state(UploadState.waiting_for_location)
        await self.state.update_data(
            text="Ховайте горіхи у дуплі старого дуба.",
            file_name="squirrel.txt",
        )

        cancel_msg = MagicMock()
        cancel_msg.text = "/cancel"
        cancel_msg.answer = AsyncMock()

        await handle_cancel_upload(message=cancel_msg, state=self.state)

        self.assertIsNone(self.state.state)
        self.assertEqual(self.state.data, {})
        cancel_msg.answer.assert_awaited_once()
        self.doc_repo.add_document.assert_not_awaited()

    def test_upload_py_contains_required_filters(self) -> None:
        """Перевірка наявності фільтрів F.text & ~F.text.startswith('/') та F.text.startswith('/')."""
        upload_file = Path(__file__).resolve().parent.parent / "bot" / "handlers" / "upload.py"
        source = upload_file.read_text(encoding="utf-8")
        self.assertIn("F.text & ~F.text.startswith('/')", source)
        self.assertIn("@router.message(UploadState.waiting_for_location, F.text.startswith('/'))", source)


if __name__ == "__main__":
    unittest.main()
