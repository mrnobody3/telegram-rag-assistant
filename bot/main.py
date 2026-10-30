"""Точка входу для запуску Telegram-бота 'Мишка-шкряботушка'."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from bot.handlers.query import router as query_router
from bot.handlers.upload import router as upload_router

try:
    from aiogram import BaseMiddleware, Bot, Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import TelegramObject
except ImportError:

    class BaseMiddleware:  # type: ignore[no-redef]
        """Запасний клас BaseMiddleware для імпорту без встановленого aiogram."""

    Bot = Any  # type: ignore[assignment,misc]
    Dispatcher = Any  # type: ignore[assignment,misc]
    MemoryStorage = Any  # type: ignore[assignment,misc]
    TelegramObject = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Налаштовує стандартне логування для застосунку."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


class DatabaseRepositoryMiddleware(BaseMiddleware):
    """Middleware для створення асинхронної сесії БД та ін'єкції DocumentRepository у хендлери."""

    def __init__(self, session_factory: Any) -> None:
        super().__init__()
        self._session_factory = session_factory

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        from bot.database.sqlite_repo import DocumentRepository

        async with self._session_factory() as session:
            data["doc_repo"] = DocumentRepository(session=session)
            return await handler(event, data)


async def main() -> None:
    """Ініціалізує інфраструктуру (SQLite, ChromaDB, Groq, AIogram) та запускає polling."""
    setup_logging()
    logger.info("Запуск Telegram-бота 'Мишка-шкряботушка'...")

    from bot.config import config
    from bot.database.chroma_repo import VectorStore
    from bot.database.session import async_session_factory, engine, init_db
    from bot.services.llm_groq import GroqService
    from bot.services.rag_engine import RAGService

    # 1. Ініціалізація реляційної БД через Base.metadata.create_all
    await init_db()

    # 2. Створення екземплярів сервісів
    vector_store = VectorStore(persist_path=config.CHROMA_PATH)
    groq_service = GroqService(
        api_key=config.GROQ_API_KEY.get_secret_value(),
        model_llm=config.GROQ_MODEL_LLM,
        model_extract=config.GROQ_MODEL_EXTRACT,
    )
    rag_service = RAGService(
        vector_store=vector_store,
        groq_service=groq_service,
    )

    # 3. Налаштування Bot та Dispatcher
    bot = Bot(token=config.BOT_TOKEN.get_secret_value())
    dp = Dispatcher(
        storage=MemoryStorage(),
        vector_store=vector_store,
        groq_service=groq_service,
        rag_service=rag_service,
    )

    # 4. Підключення middleware сесії БД та роутерів
    dp.update.middleware(DatabaseRepositoryMiddleware(async_session_factory))
    dp.include_router(upload_router)
    dp.include_router(query_router)

    # 5. Запуск polling
    try:
        logger.info("Початок опитування оновлень (start_polling)...")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await engine.dispose()
        logger.info("Бот 'Мишка-шкряботушка' коректно завершив роботу.")


if __name__ == "__main__":
    asyncio.run(main())
