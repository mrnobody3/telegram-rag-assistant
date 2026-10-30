"""Тести для сервісу парсингу файлів (патерн Strategy) та нормалізації тексту."""

import unittest

from bot.exceptions import (
    DocumentParsingError,
    UnsupportedFileFormatError,
)
from bot.services.parser import (
    DocxParser,
    ParserFactory,
    PDFParser,
    TxtParser,
    normalize_text,
)


class TestFileParser(unittest.TestCase):
    """Набір тестів для перевірки стратегій вилучення тексту, фабрики та очищення тексту."""

    def test_parser_factory_resolution(self) -> None:
        """Перевірка коректного вибору стратегій за розширенням."""
        self.assertIsInstance(ParserFactory.get_parser(".pdf"), PDFParser)
        self.assertIsInstance(ParserFactory.get_parser("PDF"), PDFParser)
        self.assertIsInstance(ParserFactory.get_parser(".docx"), DocxParser)
        self.assertIsInstance(ParserFactory.get_parser("docx"), DocxParser)
        self.assertIsInstance(ParserFactory.get_parser(".txt"), TxtParser)
        self.assertIsInstance(ParserFactory.get_parser("txt"), TxtParser)

    def test_parser_factory_unsupported_format(self) -> None:
        """Перевірка реакції на непідтримуваний формат файлу."""
        with self.assertRaises(UnsupportedFileFormatError) as ctx:
            ParserFactory.get_parser(".exe")
        self.assertIn("не підтримується", str(ctx.exception))

    def test_txt_parser_valid_utf8(self) -> None:
        """Перевірка успішного декодування тексту UTF-8."""
        parser = TxtParser()
        sample_text = "Мишка-шкряботушка знайшла зернятко коло річки."
        result = parser.extract_text(sample_text.encode("utf-8"))
        self.assertEqual(result, sample_text)

    def test_txt_parser_utf8_sig(self) -> None:
        """Перевірка успішного декодування тексту UTF-8 з BOM маркою."""
        parser = TxtParser()
        sample_text = "Запис лісової книги з BOM маркою."
        result = parser.extract_text(sample_text.encode("utf-8-sig"))
        self.assertEqual(result, sample_text)

    def test_txt_parser_empty_bytes(self) -> None:
        """Перевірка обробки порожніх байтів у TXT."""
        parser = TxtParser()
        self.assertEqual(parser.extract_text(b""), "")

    def test_txt_parser_invalid_encoding(self) -> None:
        """Перевірка викидання винятку при невалідному кодуванні."""
        parser = TxtParser()
        invalid_bytes = b"\xff\xfe\x00\x00\xaa\xbb"
        with self.assertRaises(DocumentParsingError):
            parser.extract_text(invalid_bytes)

    def test_pdf_parser_empty_bytes(self) -> None:
        """Перевірка обробки порожніх байтів у PDF."""
        parser = PDFParser()
        with self.assertRaises(DocumentParsingError):
            parser.extract_text(b"")

    def test_docx_parser_empty_bytes(self) -> None:
        """Перевірка обробки порожніх байтів у DOCX."""
        parser = DocxParser()
        with self.assertRaises(DocumentParsingError):
            parser.extract_text(b"")

    def test_normalize_text_collapses_whitespace_and_control_chars(self) -> None:
        """Чиста функція normalize_text прибирає розриви сторінок, табуляції, зайві пробіли та спецсимволи."""
        dirty_text = "  Порада\n\n\nдля   лісових\t\tзвірів\x0c\u200b\x00  у   бору. \r\n "
        self.assertEqual(normalize_text(dirty_text), "Порада для лісових звірів у бору.")

    def test_parser_from_factory_normalizes_extracted_text(self) -> None:
        """Парсер, отриманий через ParserFactory, повертає щільний нормалізований текст без зайвих пробілів."""
        parser = ParserFactory.get_parser("txt")
        dirty_bytes = "Перший рядок.\n\n\n\n   Другий\t\tрядок\x0c   третій.".encode("utf-8")
        result = parser.extract_text(dirty_bytes)
        self.assertEqual(result, "Перший рядок. Другий рядок третій.")


if __name__ == "__main__":
    unittest.main()
