"""Сервіс вилучення тексту з файлів на основі патерну Strategy."""

import io
import logging
from abc import ABC, abstractmethod

from bot.exceptions import DocumentParsingError, UnsupportedFileFormatError

logger = logging.getLogger(__name__)


class FileParser(ABC):
    """Базовий абстрактний клас стратегії вилучення тексту з файлу."""

    @abstractmethod
    def extract_text(self, file_bytes: bytes) -> str:
        """Вилучає текстовий вміст із переданого масиву байтів.

        :param file_bytes: Вміст файлу у вигляді байтів.
        :return: Видобутий очищений текст.
        :raises DocumentParsingError: Якщо виникла помилка під час парсингу байтів.
        """
        raise NotImplementedError


class PDFParser(FileParser):
    """Стратегія вилучення тексту з PDF-документів за допомогою бібліотеки pypdf."""

    def extract_text(self, file_bytes: bytes) -> str:
        """Вилучає текст з усіх сторінок PDF-файлу.

        :param file_bytes: Байти PDF-документа.
        :return: Об'єднаний текст сторінок.
        :raises DocumentParsingError: При пошкодженому або нечитабельному PDF.
        """
        if not file_bytes:
            raise DocumentParsingError("Неможливо вилучити текст з порожнього PDF файлу.")

        try:
            import pypdf
            from pypdf.errors import PyPdfError
        except ImportError as exc:
            logger.error("Бібліотека pypdf не встановлена: %s", exc)
            raise DocumentParsingError("Модуль 'pypdf' не встановлений для читання PDF файлів.") from exc

        try:
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)

            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                    if not decrypted:
                        raise DocumentParsingError("PDF-файл захищено паролем і його неможливо прочитати.")
                except Exception as exc:
                    raise DocumentParsingError(f"Помилка дешифрування PDF: {exc}") from exc

            pages_text: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted.strip())

            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                logger.warning("PDF-файл не містить текстового шару (можливо, це скановане зображення).")

            logger.info("Успішно вилучено %d символів тексту з PDF (%d сторінок)", len(full_text), len(reader.pages))
            return full_text
        except PyPdfError as exc:
            logger.error("Помилка парсингу PDF бібліотекою pypdf: %s", exc)
            raise DocumentParsingError(f"Пошкоджений або некоректний PDF-файл: {exc}") from exc
        except DocumentParsingError:
            raise
        except Exception as exc:
            logger.error("Непередбачена помилка під час читання PDF: %s", exc)
            raise DocumentParsingError(f"Помилка обробки PDF-файлу: {exc}") from exc


class DocxParser(FileParser):
    """Стратегія вилучення тексту з Word-документів (.docx) за допомогою python-docx."""

    def extract_text(self, file_bytes: bytes) -> str:
        """Вилучає текст із параграфів та таблиць docx-документа.

        :param file_bytes: Байти DOCX-документа.
        :return: Об'єднаний текст документа.
        :raises DocumentParsingError: При пошкодженому архіві чи структурі DOCX.
        """
        if not file_bytes:
            raise DocumentParsingError("Неможливо вилучити текст з порожнього DOCX файлу.")

        try:
            import docx
        except ImportError as exc:
            logger.error("Бібліотека python-docx не встановлена: %s", exc)
            raise DocumentParsingError("Модуль 'python-docx' не встановлений для читання DOCX файлів.") from exc

        try:
            stream = io.BytesIO(file_bytes)
            doc = docx.Document(stream)

            text_blocks: list[str] = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    text_blocks.append(text)

            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        text_blocks.append(" | ".join(row_cells))

            full_text = "\n\n".join(text_blocks).strip()
            logger.info("Успішно вилучено %d символів тексту з DOCX", len(full_text))
            return full_text
        except Exception as exc:
            logger.error("Помилка читання DOCX файлу: %s", exc)
            raise DocumentParsingError(f"Некоректний або пошкоджений DOCX-файл: {exc}") from exc


class TxtParser(FileParser):
    """Стратегія вилучення тексту з текстових файлів (.txt) у кодуванні UTF-8."""

    def extract_text(self, file_bytes: bytes) -> str:
        """Декодує байти у форматі UTF-8 (підтримує також UTF-8 з BOM маркою).

        :param file_bytes: Байти TXT-документа.
        :return: Декодований текст.
        :raises DocumentParsingError: Якщо кодування файлу не є UTF-8.
        """
        if not file_bytes:
            return ""

        try:
            return file_bytes.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            logger.error("Помилка декодування TXT файлу: не UTF-8 (%s)", exc)
            raise DocumentParsingError(
                "Файл не вдалося прочитати у кодуванні UTF-8. Будь ласка, збережіть файл у кодуванні UTF-8."
            ) from exc


class ParserFactory:
    """Фабрика для вибору відповідної стратегії парсингу за розширенням файлу."""

    _registry: dict[str, type[FileParser]] = {
        "pdf": PDFParser,
        "docx": DocxParser,
        "txt": TxtParser,
    }

    @classmethod
    def get_parser(cls, extension: str) -> FileParser:
        """Повертає екземпляр FileParser для вказаного розширення файлу.

        :param extension: Розширення файлу (наприклад, 'pdf', '.docx', 'TXT').
        :return: Екземпляр відповідного класу парсера.
        :raises UnsupportedFileFormatError: Якщо для розширення відсутній парсер.
        """
        normalized_ext = extension.strip().lower().lstrip(".")
        parser_cls = cls._registry.get(normalized_ext)

        if not parser_cls:
            supported = ", ".join(f".{ext}" for ext in sorted(cls._registry.keys()))
            logger.warning("Спроба отримати парсер для непідтримуваного розширення: '%s'", extension)
            raise UnsupportedFileFormatError(
                f"Формат файлу '.{normalized_ext}' не підтримується. "
                f"Підтримувані формати: {supported}."
            )

        return parser_cls()

    @classmethod
    def register_parser(cls, extension: str, parser_cls: type[FileParser]) -> None:
        """Реєструє нову стратегію парсингу для вказаного розширення.

        :param extension: Розширення файлу (наприклад, 'md').
        :param parser_cls: Клас парсера, що успадковує FileParser.
        """
        normalized_ext = extension.strip().lower().lstrip(".")
        cls._registry[normalized_ext] = parser_cls
        logger.info("Зареєстровано новий парсер %s для розширення .%s", parser_cls.__name__, normalized_ext)
