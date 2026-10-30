"""Сервіс RAG (Retrieval-Augmented Generation), що поєднує VectorStore, FastEmbed та GroqService."""

import asyncio
import logging
from typing import Any

from bot.database.chroma_repo import RelevantChunk, VectorStore
from bot.exceptions import VectorStoreError
from bot.services.llm_groq import GroqService

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Чиста функція розбиття тексту на чанки заданого розміру з перекриттям.

    :param text: Вхідний текст документа.
    :param chunk_size: Максимальна довжина одного чанка (за замовчуванням ~500 символів).
    :param overlap: Кількість символів перекриття між сусідніми чанками.
    :return: Список текстових чанків.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size повинен бути більшим за 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap повинен бути в межах [0, chunk_size).")

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += step

    return chunks


def _chunks_to_context_dicts(chunks: list[RelevantChunk]) -> list[dict[str, Any]]:
    """Чиста функція перетворення RelevantChunk у словники для GroqService.generate_answer."""
    context_list: list[dict[str, Any]] = []
    for item in chunks:
        meta = dict(item.metadata) if item.metadata else {}
        context_list.append(
            {
                "text": item.text,
                "doc_id": item.doc_id,
                "location": item.location,
                "file_name": meta.get("file_name") or meta.get("source") or f"doc_{item.doc_id}",
                "author_name": meta.get("author_name"),
                "distance": item.distance,
            }
        )
    return context_list


def _extract_unique_doc_ids(chunks: list[RelevantChunk]) -> list[int]:
    """Чиста функція отримання унікальних doc_id зі списку чанків зі збереженням порядку."""
    seen: set[int] = set()
    doc_ids: list[int] = []
    for chunk in chunks:
        if chunk.doc_id and chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            doc_ids.append(chunk.doc_id)
    return doc_ids


class RAGService:
    """Оркестратор RAG-пайплайну: чанкінг, ембединги (FastEmbed), векторний пошук та генерація відповіді."""

    def __init__(
        self,
        vector_store: VectorStore,
        groq_service: GroqService,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Any | None = None,
    ) -> None:
        """Ініціалізує RAGService із залежностями VectorStore та GroqService.

        :param vector_store: Екземпляр векторного сховища ChromaDB.
        :param groq_service: Екземпляр сервісу Groq LLM.
        :param embedding_model_name: Назва моделі для FastEmbed TextEmbedding.
        :param embedder: Ін'єкція екземпляра TextEmbedding (для тестування).
        """
        self._vector_store = vector_store
        self._groq_service = groq_service
        self._embedding_model_name = embedding_model_name
        self._embedder = embedder
        self.last_used_doc_ids: list[int] = []

    def _get_embedder(self) -> Any:
        """Повертає або ледаче ініціалізує модель fastembed.TextEmbedding."""
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                logger.error("Бібліотека fastembed не встановлена: %s", exc)
                raise VectorStoreError("Модуль 'fastembed' не встановлений.") from exc

            logger.info("Ініціалізація FastEmbed у RAGService: %s", self._embedding_model_name)
            self._embedder = TextEmbedding(model_name=self._embedding_model_name)
        return self._embedder

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Синхронно перетворює список текстів у вектори через fastembed (TextEmbedding)."""
        embedder = self._get_embedder()
        raw_embeddings = embedder.embed(texts)
        return [
            vec.tolist() if hasattr(vec, "tolist") else list(vec)
            for vec in raw_embeddings
        ]

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Асинхронно обчислює ембединги у фоновому потоці."""
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    async def process_and_save_document(
        self,
        text: str,
        doc_id: int,
        location: str = "general",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Розбиває текст на чанки, генерує вектори через FastEmbed і зберігає їх у VectorStore.

        :param text: Повний текст документа.
        :param doc_id: Ідентифікатор документа в SQLite.
        :param location: Локація лісу (за замовчуванням 'general').
        :param extra_metadata: Додаткові метадані для кожного чанка (наприклад, file_name, author_name).
        :return: Список збережених ідентифікаторів чанків.
        """
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        if not chunks:
            logger.warning("Документ doc_id=%d не містить тексту для індексації.", doc_id)
            return []

        logger.info("Генерація векторів для %d чанків документа doc_id=%d...", len(chunks), doc_id)
        embeddings = await self._embed_texts(chunks)

        extra_metadatas = (
            [dict(extra_metadata) for _ in chunks]
            if extra_metadata
            else None
        )

        return await self._vector_store.add_chunks(
            chunks=chunks,
            doc_id=doc_id,
            location=location,
            is_active=True,
            extra_metadatas=extra_metadatas,
            embeddings=embeddings,
        )

    async def answer_question(
        self,
        query: str,
        location_filter: str | None = None,
    ) -> str:
        """Перетворює запит у вектор, шукає топ-3 чанки у VectorStore та генерує відповідь через GroqService.

        :param query: Запитання користувача.
        :param location_filter: Опціональний фільтр за локацією лісу.
        :return: Згенерована відповідь від LLM на основі знайденого контексту.
        """
        cleaned_query = query.strip()
        logger.info(
            "Обробка RAG-запиту: query='%s', location_filter='%s'",
            cleaned_query,
            location_filter,
        )

        query_vectors = await self._embed_texts([cleaned_query])
        query_embedding = query_vectors[0]

        relevant_chunks = await self._vector_store.search(
            query=cleaned_query,
            location=location_filter,
            top_k=3,
            is_active=True,
            query_embedding=query_embedding,
        )

        self.last_used_doc_ids = _extract_unique_doc_ids(relevant_chunks)
        context_dicts = _chunks_to_context_dicts(relevant_chunks)
        return await self._groq_service.generate_answer(
            question=cleaned_query,
            context_chunks=context_dicts,
        )
