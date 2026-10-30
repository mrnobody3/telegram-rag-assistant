"""Репозиторій для роботи з документами у реляційній базі даних SQLite."""

import logging
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.models import Document
from bot.exceptions import DatabaseError, DocumentAlreadyExistsError

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Репозиторій для виконання CRUD-операцій над документами лісової бази знань."""

    def __init__(self, session: AsyncSession) -> None:
        """Ініціалізує репозиторій з наданою асинхронною сесією.

        :param session: Активна асинхронна сесія SQLAlchemy.
        """
        self._session = session

    async def add_document(
            self,
            file_name: str,
            location: str,
            author_id: str,
            author_name: str,
    ) -> Document:
        """Створює новий запис документа в базі даних.

        :param file_name: Назва завантаженого файлу.
        :param location: Локація лісу.
        :param author_id: Ідентифікатор автора в Telegram.
        :param author_name: Ім'я автора.
        :return: Створений об'єкт Document.
        :raises DocumentAlreadyExistsError: Якщо порушено обмеження унікальності.
        :raises DatabaseError: При інших помилках взаємодії з БД.
        """
        logger.info(
            "Додавання нового документа: file_name='%s', location='%s', author='%s' (%s)",
            file_name,
            location,
            author_name,
            author_id,
        )
        document = Document(
            file_name=file_name,
            location=location,
            author_id=author_id,
            author_name=author_name,
            is_active=True,
        )

        try:
            self._session.add(document)
            await self._session.commit()
            await self._session.refresh(document)
            logger.info("Документ успішно збережено з ID=%d", document.id)
            return document
        except IntegrityError as exc:
            await self._session.rollback()
            logger.warning(
                "Конфлікт унікальності для документа '%s' у локації '%s': %s",
                file_name,
                location,
                exc,
            )
            raise DocumentAlreadyExistsError(
                f"Документ з назвою '{file_name}' для локації '{location}' вже існує."
            ) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.error("Помилка SQLAlchemy під час збереження документа: %s", exc)
            raise DatabaseError(f"Помилка збереження документа в БД: {exc}") from exc

    async def deactivate_old_versions(self, file_name: str, location: str) -> int:
        """Деактивує (встановлює is_active=False) попередні версії документа.

        :param file_name: Назва файлу документа.
        :param location: Локація лісу.
        :return: Кількість оновлених (деактивованих) документів.
        :raises DatabaseError: При помилках виконання запиту в БД.
        """
        logger.info(
            "Деактивація старих версій для file_name='%s', location='%s'",
            file_name,
            location,
        )
        stmt = (
            update(Document)
            .where(
                Document.file_name == file_name,
                Document.location == location,
                Document.is_active.is_(True),
            )
            .values(is_active=False)
        )

        try:
            result = await self._session.execute(stmt)
            await self._session.commit()
            deactivated_count = int(result.rowcount or 0)
            logger.info("Деактивовано старих версій документа: %d", deactivated_count)
            return deactivated_count
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.error("Помилка деактивації старих версій документа: %s", exc)
            raise DatabaseError(f"Помилка деактивації версій: {exc}") from exc

    async def get_by_id(self, doc_id: int) -> Document | None:
        """Отримує документ за його ідентифікатором.

        :param doc_id: Унікальний ID документа.
        :return: Знайдений Document або None.
        """
        stmt = select(Document).where(Document.id == doc_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_by_name_and_location(
            self,
            file_name: str,
            location: str,
    ) -> Document | None:
        """Отримує активний документ за назвою файлу та локацією.

        :param file_name: Назва файлу.
        :param location: Локація лісу.
        :return: Активний Document або None.
        """
        stmt = select(Document).where(
            Document.file_name == file_name,
            Document.location == location,
            Document.is_active.is_(True),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active_by_location(self, location: str) -> Sequence[Document]:
        """Отримує список усіх активних документів для зазначеної локації.

        :param location: Локація лісу.
        :return: Послідовність активних документів.
        """
        stmt = (
            select(Document)
            .where(
                Document.location == location,
                Document.is_active.is_(True),
            )
            .order_by(Document.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()
