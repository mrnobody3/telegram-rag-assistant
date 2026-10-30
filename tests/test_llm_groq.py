"""Тести для сервісу GroqService (вилучення локації та генерація відповідей)."""

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from bot.exceptions import LLMServiceError
from bot.services.llm_groq import GroqService


def _make_mock_response(content: str | None) -> MagicMock:
    """Створює мок-відповідь від клієнта Groq chat.completions.create."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestGroqService(unittest.IsolatedAsyncioTestCase):
    """Набір асинхронних тестів для перевірки поведінки GroqService."""

    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock()
        self.service = GroqService(
            api_key="gsk_test_key",
            model_llm="openai/gpt-oss-120b",
            model_extract="openai/gpt-oss-20b",
            client=self.mock_client,
        )

    async def test_extract_location_returns_string(self) -> None:
        """Перевірка успішного вилучення локації з JSON-відповіді."""
        self.mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"location": "хвойний ліс"}'
        )

        result = await self.service.extract_location("У сосновому бору ростуть білі гриби.")
        self.assertEqual(result, "хвойний ліс")

        call_kwargs: dict[str, Any] = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "openai/gpt-oss-20b")
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})

    async def test_extract_location_returns_none_when_null(self) -> None:
        """Перевірка повернення None, якщо в JSON location дорівнює null."""
        self.mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"location": null}'
        )

        result = await self.service.extract_location("Загальна порада для всіх звірів.")
        self.assertIsNone(result)

    async def test_extract_location_empty_input(self) -> None:
        """Порожній вхідний текст не повинен робити зайвий API-виклик."""
        result = await self.service.extract_location("   ")
        self.assertIsNone(result)
        self.mock_client.chat.completions.create.assert_not_called()

    async def test_extract_location_invalid_json_raises_error(self) -> None:
        """Некоректний JSON від моделі викликає LLMServiceError."""
        self.mock_client.chat.completions.create.return_value = _make_mock_response(
            "not a json"
        )

        with self.assertRaises(LLMServiceError):
            await self.service.extract_location("Текст про болото")

    async def test_generate_answer_returns_text(self) -> None:
        """Перевірка генерації відповіді на основі чанків контексту."""
        expected_answer = "Збирайте жолуді під старими дубами. Джерело: oak_guide.pdf (Автор: Сова)."
        self.mock_client.chat.completions.create.return_value = _make_mock_response(
            expected_answer
        )

        chunks = [
            {
                "text": "Найкращі жолуді падають у жовтні під старими дубами.",
                "file_name": "oak_guide.pdf",
                "author_name": "Сова",
                "location": "дубовий гай",
            }
        ]
        answer = await self.service.generate_answer("Де шукати жолуді?", chunks)
        self.assertEqual(answer, expected_answer)

        call_kwargs: dict[str, Any] = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "openai/gpt-oss-120b")
        messages = call_kwargs["messages"]
        self.assertTrue(any("Ти лісовий помічник" in m["content"] for m in messages))
        self.assertTrue(any("oak_guide.pdf" in m["content"] for m in messages))

    async def test_generate_answer_empty_question_raises_error(self) -> None:
        """Порожнє запитання викликає LLMServiceError."""
        with self.assertRaises(LLMServiceError):
            await self.service.generate_answer("   ", [{"text": "Контекст"}])

    async def test_model_not_found_falls_back_to_available_model(self) -> None:
        """Якщо вказана в конфігу модель повертає model_not_found (404), сервіс автоматично використовує резервну модель."""
        legacy_service = GroqService(
            api_key="gsk_test_key",
            model_llm="llama-3.3-70b-versatile",
            model_extract="llama-3.1-8b-instant",
            client=self.mock_client,
        )
        self.mock_client.chat.completions.create.side_effect = [
            Exception("Error code: 404 - {'error': {'message': 'The model `llama-3.1-8b-instant` does not exist', 'code': 'model_not_found'}}"),
            _make_mock_response('{"location": "березовий гай"}'),
        ]

        result = await legacy_service.extract_location("У березовому гаю ростуть підберезники.")
        self.assertEqual(result, "березовий гай")
        self.assertEqual(self.mock_client.chat.completions.create.await_count, 2)
        second_call_model = self.mock_client.chat.completions.create.call_args_list[1].kwargs["model"]
        self.assertIn(second_call_model, ("openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"))


if __name__ == "__main__":
    unittest.main()
