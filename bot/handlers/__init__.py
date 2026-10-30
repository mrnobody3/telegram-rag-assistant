"""Обробники подій (handlers) Telegram-бота."""

from bot.handlers.query import router as query_router
from bot.handlers.upload import router as upload_router

__all__ = ["query_router", "upload_router"]
