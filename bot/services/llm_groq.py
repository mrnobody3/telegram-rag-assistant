"""Сервіс для асинхронної взаємодії з Groq API (екстракція метаданих та генерація RAG-відповідей)."""

import json
import logging
from typing import Any

from bot.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

EXTRACT_LOCATION_SYSTEM_PROMPT = (
    "Проаналізуй текст. Визнач локацію або місце, для якого ця порада "
    "(наприклад, 'хвойний ліс'). Якщо не вказано - null. "
    'Поверни суворий JSON {"location": string | null}'
)

GENERATE_ANSWER_SYSTEM_PROMPT = (
    "Ти лісовий помічник. Дай пораду на основі наданого контексту. "
    "Обов'язково вкажи джерело"
)

FALLBACK_EXTRACT_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
)

FALLBACK_LLM_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
)


def _is_model_not_found_error(exc: Exception) -> bool:
    """Чиста функція перевірки, чи викликана помилка відсутністю моделі (404 / model_not_found)."""
    err_text = str(exc).lower()
    return (
        "model_not_found" in err_text
        or "does not exist" in err_text
        or "model_decommissioned" in err_text
    )


def _parse_location_json(raw_content: str) -> str | None:
    """Чиста функція розбору JSON-відповіді з локацією на межі системи."""
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(f"Модель повернула некоректний JSON: {raw_content}") from exc

    if not isinstance(payload, dict):
        raise LLMServiceError(f"Очікувався JSON-об'єкт, отримано: {type(payload).__name__}")

    location = payload.get("location")
    if location is None:
        return None

    normalized = str(location).strip()
    if not normalized or normalized.lower() == "null":
        return None

    return normalized


def _format_context_chunks(context_chunks: list[dict[str, Any]]) -> str:
    """Чиста функція форматування контекстних чанків разом із метаданими джерела."""
    if not context_chunks:
        return "Контекст відсутній."

    formatted_blocks: list[str] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        text = str(chunk.get("text") or chunk.get("document") or "").strip()
        source = (
            chunk.get("source")
            or chunk.get("file_name")
            or chunk.get("doc_id")
            or "Невідоме джерело"
        )
        author = chunk.get("author_name") or chunk.get("author")
        location = chunk.get("location")

        meta_parts = [f"Джерело: {source}"]
        if author:
            meta_parts.append(f"Автор: {author}")
        if location:
            meta_parts.append(f"Локація: {location}")

        header = f"[Фрагмент #{idx} | {', '.join(meta_parts)}]"
        formatted_blocks.append(f"{header}\n{text}")

    return "\n\n".join(formatted_blocks)


class GroqService:
    """Сервіс інтеграції з Groq API для вилучення локації та генерації відповідей."""

    def __init__(
        self,
        api_key: str | None = None,
        model_llm: str | None = None,
        model_extract: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Ініціалізує асинхронний клієнт Groq та конфігурацію моделей.

        :param api_key: API-ключ Groq (за замовчуванням з config.GROQ_API_KEY).
        :param model_llm: Основна генеративна модель (за замовчуванням config.GROQ_MODEL_LLM).
        :param model_extract: Модель для екстракції (за замовчуванням config.GROQ_MODEL_EXTRACT).
        :param client: Готовий екземпляр AsyncGroq (для ін'єкції залежностей та тестування).
        """
        if client is not None:
            self._client = client
            self._api_key = api_key or ""
            self._model_llm = model_llm or "openai/gpt-oss-120b"
            self._model_extract = model_extract or "openai/gpt-oss-20b"
        else:
            from bot.config import config

            self._api_key = api_key or config.GROQ_API_KEY.get_secret_value()
            self._model_llm = model_llm or config.GROQ_MODEL_LLM
            self._model_extract = model_extract or config.GROQ_MODEL_EXTRACT
            self._client = self._create_async_client(self._api_key)

    @staticmethod
    def _create_async_client(api_key: str) -> Any:
        """Створює екземпляр AsyncGroq."""
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            logger.error("Бібліотека groq не встановлена: %s", exc)
            raise LLMServiceError("Модуль 'groq' не встановлений.") from exc

        return AsyncGroq(api_key=api_key)

    async def _create_completion_with_fallback(
        self,
        primary_model: str,
        fallback_models: tuple[str, ...],
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Виконує chat.completions.create із автоматичним перемиканням на доступні моделі при 404."""
        candidates: list[str] = [primary_model]
        for candidate in fallback_models:
            if candidate not in candidates:
                candidates.append(candidate)

        last_exc: Exception | None = None
        for model_name in candidates:
            try:
                response = await self._client.chat.completions.create(
                    model=model_name,
                    **kwargs,
                )
                return response, model_name
            except Exception as exc:
                last_exc = exc
                if _is_model_not_found_error(exc):
                    logger.warning(
                        "Модель '%s' недоступна у Groq API (model_not_found), спроба резервної моделі...",
                        model_name,
                    )
                    continue
                raise

        raise last_exc or RuntimeError("Не вдалося виконати запит до жодної моделі Groq.")

    async def extract_location(self, text_sample: str) -> str | None:
        """Визначає локацію лісу з фрагмента тексту за допомогою JSON mode.

        :param text_sample: Фрагмент тексту документа.
        :return: Назва локації або None, якщо локацію не вказано.
        :raises LLMServiceError: При помилці звернення до Groq API або невалідному JSON.
        """
        cleaned_sample = text_sample.strip()
        if not cleaned_sample:
            return None

        logger.info("Вилучення локації через модель %s...", self._model_extract)
        try:
            response, used_model = await self._create_completion_with_fallback(
                primary_model=self._model_extract,
                fallback_models=FALLBACK_EXTRACT_MODELS,
                messages=[
                    {"role": "system", "content": EXTRACT_LOCATION_SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned_sample},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            self._model_extract = used_model
            raw_content = response.choices[0].message.content
            if not raw_content:
                return None

            location = _parse_location_json(raw_content)
            logger.info("Визначена локація: %s", location)
            return location
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.error("Помилка під час виклику Groq API (extract_location): %s", exc)
            raise LLMServiceError(f"Помилка визначення локації через Groq API: {exc}") from exc

    async def generate_answer(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Генерує пораду лісового помічника на основі наданих контекстних чанків.

        :param question: Запитання користувача.
        :param context_chunks: Список словників із текстом та метаданими джерел.
        :return: Згенерована відповідь із вказанням джерела.
        :raises LLMServiceError: При порожньому запитанні або помилці API.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise LLMServiceError("Запитання не може бути порожнім.")

        formatted_context = _format_context_chunks(context_chunks)
        user_prompt = (
            f"Контекст:\n{formatted_context}\n\n"
            f"Запитання: {cleaned_question}"
        )

        logger.info(
            "Генерація відповіді через модель %s (чанків контексту: %d)...",
            self._model_llm,
            len(context_chunks),
        )
        try:
            response, used_model = await self._create_completion_with_fallback(
                primary_model=self._model_llm,
                fallback_models=FALLBACK_LLM_MODELS,
                messages=[
                    {"role": "system", "content": GENERATE_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            self._model_llm = used_model
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise LLMServiceError("Модель Groq повернула порожню відповідь.")

            return answer
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.error("Помилка під час виклику Groq API (generate_answer): %s", exc)
            raise LLMServiceError(f"Помилка генерації відповіді через Groq API: {exc}") from exc
