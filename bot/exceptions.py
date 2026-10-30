"""Базові та специфічні винятки застосунку."""

RATE_LIMIT_USER_MESSAGE = (
    "Зараз багато звірів читають документи, зачекайте хвилинку і спробуйте знову"
)


class AppException(Exception):
    """Базовий виняток для всіх помилок застосунку."""

    def __init__(self, message: str = "Виникла невідома помилка застосунку.") -> None:
        super().__init__(message)
        self.message = message


class DatabaseError(AppException):
    """Помилка взаємодії з реляційною базою даних."""


class DocumentNotFoundError(DatabaseError):
    """Виникає, коли запитаний документ не знайдено в БД."""


class DocumentAlreadyExistsError(DatabaseError):
    """Виникає при порушенні унікальності документу."""


class VectorStoreError(AppException):
    """Помилка взаємодії з векторним сховищем (ChromaDB)."""


class DocumentParsingError(AppException):
    """Помилка парсингу або обробки файлу документу."""


class UnsupportedFileFormatError(DocumentParsingError):
    """Виникає, коли формат завантаженого файлу не підтримується."""


class LLMServiceError(AppException):
    """Помилка взаємодії з API мовної моделі (Groq)."""


class RateLimitExceededError(LLMServiceError):
    """Виникає при перевищенні лімітів запитів або токенів (HTTP 429 Too Many Requests)."""

    def __init__(self, message: str = RATE_LIMIT_USER_MESSAGE) -> None:
        super().__init__(message)
