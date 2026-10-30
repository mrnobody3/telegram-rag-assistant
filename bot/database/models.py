"""Декларативні моделі SQLAlchemy 2.0 для реляційної бази даних."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String, UniqueConstraint, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовий декларативний клас SQLAlchemy."""


class Document(Base):
    """Модель збереженого документа лісових знань.

    Зберігає метадані завантаженого файлу, автора та прив'язку до локації лісу.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
        doc="Унікальний ідентифікатор документа",
    )
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Назва файлу документа",
    )
    location: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Локація лісу, до якої належить документ (наприклад, 'Дубовий гай')",
    )
    author_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Telegram ID або унікальний ідентифікатор автора",
    )
    author_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Ім'я або псевдонім звіра-автора",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        doc="Час створення/завантаження запису",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("1"),
        nullable=False,
        index=True,
        doc="Ознака актуальності документа (активна версія)",
    )

    __table_args__ = (
        UniqueConstraint(
            "file_name",
            "location",
            name="uq_document_file_name_location",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, file_name='{self.file_name}', "
            f"location='{self.location}', is_active={self.is_active})>"
        )
