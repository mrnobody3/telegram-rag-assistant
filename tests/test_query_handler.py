"""Тести для обробника запитань користувачів (bot/handlers/query.py) та форматування джерел."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from bot.handlers.query import format_answer_with_sources, handle_text_query


class TestQueryHandler(unittest.IsolatedAsyncioTestCase):
    """Тести для обробки текстових запитань звірів та додавання метаданих джерел."""

    def test_format_answer_with_sources_pure(self) -> None:
        """Перевірка чистої функції форматування відповіді з автором і датою."""
        doc = MagicMock()
        doc.id = 5
        doc.file_name = "berries.pdf"
        doc.author_name = "Лисичка"
        doc.created_at = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

        formatted = format_answer_with_sources(
            answer="Малину краще збирати на сонячних галявинах.",
            documents=[doc],
        )

        self.assertIn("Малину краще збирати на сонячних галявинах.", formatted)
        self.assertIn("Лисичка", formatted)
        self.assertIn("2026-10-30", formatted)
        self.assertIn("berries.pdf", formatted)

    async def test_handle_text_query_fetches_docs_and_replies(self) -> None:
        """Обробник викликає RAGService, дістає документи за doc_id з репозиторію і відповідає."""
        message = MagicMock()
        message.text = "Де шукати малину?"
        message.answer = AsyncMock()

        doc = MagicMock()
        doc.id = 5
        doc.file_name = "berries.pdf"
        doc.author_name = "Лисичка"
        doc.created_at = datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)

        doc_repo = MagicMock()
        doc_repo.get_by_id = AsyncMock(return_value=doc)

        rag_service = MagicMock()
        rag_service.answer_question = AsyncMock(
            return_value="Шукайте на галявині біля струмка."
        )
        rag_service.last_used_doc_ids = [5]

        await handle_text_query(
            message=message,
            rag_service=rag_service,
            doc_repo=doc_repo,
        )

        rag_service.answer_question.assert_awaited_once_with("Де шукати малину?", location_filter=None)
        doc_repo.get_by_id.assert_awaited_once_with(5)
        message.answer.assert_awaited_once()
        reply_text = message.answer.call_args.args[0]
        self.assertIn("Шукайте на галявині біля струмка.", reply_text)
        self.assertIn("Лисичка", reply_text)
        self.assertIn("2026-10-30", reply_text)


if __name__ == "__main__":
    unittest.main()
