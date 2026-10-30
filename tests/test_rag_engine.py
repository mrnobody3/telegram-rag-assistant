"""Тести для RAGService та функції розбиття тексту на чанки (chunk_text)."""

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from bot.services.rag_engine import RAGService, chunk_text


class FakeEmbedder:
    """Тестовий ембедер, що імітує поведінку fastembed.TextEmbedding.embed."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.5, 0.1] for t in texts]


class TestChunkText(unittest.TestCase):
    """Тести чистої функції розбиття тексту chunk_text."""

    def test_empty_text_returns_empty_list(self) -> None:
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Коротка порада для їжачка."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(chunks, [text])

    def test_long_text_chunks_with_overlap(self) -> None:
        text = "АБВГҐДЕЄЖЗ" * 120  # 1200 символів
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        # Перевірка перекриття: кінець першого чанка збігається з початком другого
        self.assertEqual(chunks[0][-100:], chunks[1][:100])


class TestRAGService(unittest.IsolatedAsyncioTestCase):
    """Асинхронні тести для перевірки RAGService."""

    def setUp(self) -> None:
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.add_chunks = AsyncMock(return_value=["doc_7_chunk_0", "doc_7_chunk_1"])
        self.mock_vector_store.search = AsyncMock()

        self.mock_groq_service = MagicMock()
        self.mock_groq_service.generate_answer = AsyncMock(
            return_value="Збирайте шишки на світанку. Джерело: doc_7."
        )

        self.fake_embedder = FakeEmbedder()
        self.rag_service = RAGService(
            vector_store=self.mock_vector_store,
            groq_service=self.mock_groq_service,
            embedder=self.fake_embedder,
        )

    async def test_process_and_save_document(self) -> None:
        """Перевірка розбиття тексту на чанки, генерації векторів та збереження у VectorStore."""
        text = "Порада для білочки. " * 40  # ~800 символів -> 2 чанки
        chunk_ids = await self.rag_service.process_and_save_document(
            text=text,
            doc_id=7,
            location="хвойний ліс",
        )

        self.assertEqual(chunk_ids, ["doc_7_chunk_0", "doc_7_chunk_1"])
        self.mock_vector_store.add_chunks.assert_awaited_once()
        call_kwargs: dict[str, Any] = self.mock_vector_store.add_chunks.call_args.kwargs
        self.assertEqual(call_kwargs["doc_id"], 7)
        self.assertEqual(call_kwargs["location"], "хвойний ліс")
        self.assertEqual(len(call_kwargs["chunks"]), 2)
        self.assertEqual(len(call_kwargs["embeddings"]), 2)

    async def test_answer_question_searches_top_3_and_calls_llm(self) -> None:
        """Перевірка векторизації запиту, пошуку топ-3 чанків та генерації відповіді."""
        mock_chunk = MagicMock()
        mock_chunk.text = "Шишки найкраще збирати на світанку."
        mock_chunk.doc_id = 7
        mock_chunk.location = "хвойний ліс"
        mock_chunk.metadata = {"doc_id": 7, "location": "хвойний ліс", "file_name": "pine.txt"}
        self.mock_vector_store.search.return_value = [mock_chunk]

        answer = await self.rag_service.answer_question(
            query="Коли збирати шишки?",
            location_filter="хвойний ліс",
        )

        self.assertEqual(answer, "Збирайте шишки на світанку. Джерело: doc_7.")
        self.mock_vector_store.search.assert_awaited_once()
        search_kwargs: dict[str, Any] = self.mock_vector_store.search.call_args.kwargs
        self.assertEqual(search_kwargs["top_k"], 3)
        self.assertEqual(search_kwargs["location"], "хвойний ліс")
        self.assertIsNotNone(search_kwargs.get("query_embedding"))

        self.mock_groq_service.generate_answer.assert_awaited_once()
        llm_args = self.mock_groq_service.generate_answer.call_args
        passed_chunks = llm_args.kwargs.get("context_chunks") or llm_args.args[1]
        self.assertEqual(len(passed_chunks), 1)
        self.assertEqual(passed_chunks[0]["text"], "Шишки найкраще збирати на світанку.")


if __name__ == "__main__":
    unittest.main()
