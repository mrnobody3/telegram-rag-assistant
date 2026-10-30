"""Керування асинхронними сесіями та підключенням до SQLite."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bot.config import config
from bot.database.models import Base

logger = logging.getLogger(__name__)

engine: AsyncEngine = create_async_engine(
    config.sqlite_url,
    echo=False,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Створює всі необхідні таблиці в базі даних, якщо вони ще не існують."""
    config.ensure_directories()
    logger.info("Ініціалізація структури бази даних...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Структура бази даних успішно ініціалізована.")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронний контекстний менеджер для отримання сесії БД."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
