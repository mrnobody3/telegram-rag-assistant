"""Векторне сховище на базі ChromaDB та FastEmbed для семантичного пошуку."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class RelevantChunk:
    """Модель знайденого релевантного текстового чанка."""

    chunk_id: str
    text: str
    doc_id: int
    location: str
    distance: float
    metadata: dict[str, Any]


class VectorStore:
    """Клас-обгортка для роботи з векторною БД chromadb.PersistentClient."""

    def __init__(
            self,
            persist_path: str | Path | None = None,
            collection_name: str = "forest_knowledge",
            embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            client: Any | None = None,
    ) -> None:
        """Ініціалізує клієнт ChromaDB та налаштування моделі ембедингів.

        :param persist_path: Шлях для збереження векторної БД (за замовчуванням config.CHROMA_PATH).
        :param collection_name: Назва колекції чанків лісових знань.
        :param embedding_model_name: Назва моделі FastEmbed (підтримує українську мову).
        :param client: Готовий клієнт ChromaDB (для тестування).
        """
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name
        self._embedder: Any = None
        self._client: Any

        if client is not None:
            self._persist_path = Path(persist_path or "data/chroma_data")
            self._client = client
        else:
            import chromadb
            from bot.config import config

            self._persist_path = Path(persist_path or config.CHROMA_PATH)
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_path.resolve())
            )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedder(self) -> Any:
        """Повертає або ледаче ініціалізує модель ембедингів FastEmbed."""
        if self._embedder is None:
            from fastembed import TextEmbedding

            logger.info("Ініціалізація моделі ембедингів FastEmbed: %s", self._embedding_model_name)
            self._embedder = TextEmbedding(model_name=self._embedding_model_name)
        return self._embedder

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Генерує векторні представлення для списку текстів через FastEmbed."""
        embedder = self._get_embedder()
        embeddings_iter = embedder.embed(texts)
        return [
            emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for emb in embeddings_iter
        ]

    def _add_chunks_sync(
            self,
            chunks: list[str],
            doc_id: int,
            location: str = "general",
            is_active: bool = True,
            extra_metadatas: list[dict[str, Any]] | None = None,
            embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Синхронне додавання чанків у ChromaDB."""
        if not chunks:
            logger.warning("Спроба додати порожній список чанків для doc_id=%d", doc_id)
            return []

        logger.info(
            "Додавання %d чанків для doc_id=%d, location='%s'",
            len(chunks),
            doc_id,
            location,
        )

        try:
            final_embeddings = (
                embeddings if embeddings is not None else self._generate_embeddings(chunks)
            )
            chunk_ids: list[str] = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]

            metadatas: list[dict[str, Any]] = []
            for i in range(len(chunks)):
                meta: dict[str, Any] = {
                    "doc_id": doc_id,
                    "location": location,
                    "is_active": is_active,
                    "chunk_index": i,
                }
                if extra_metadatas and i < len(extra_metadatas):
                    meta.update(extra_metadatas[i])
                metadatas.append(meta)

            self._collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=final_embeddings,
                metadatas=metadatas,
            )
            logger.info("Успішно збережено %d чанків у ChromaDB", len(chunk_ids))
            return chunk_ids
        except Exception as exc:
            logger.error("Помилка додавання чанків у ChromaDB: %s", exc)
            raise VectorStoreError(f"Не вдалося додати чанки в ChromaDB: {exc}") from exc

    def _search_sync(
            self,
            query: str,
            location: str | None = None,
            top_k: int = 5,
            is_active: bool = True,
            query_embedding: list[float] | None = None,
    ) -> list[RelevantChunk]:
        """Синхронний пошук релевантних чанків з фільтрацією."""
        logger.info(
            "Пошук у ChromaDB: query='%s', location='%s', is_active=%s, top_k=%d",
            query,
            location,
            is_active,
            top_k,
        )
        try:
            query_embeddings = (
                [query_embedding]
                if query_embedding is not None
                else self._generate_embeddings([query])
            )

            if location:
                where_filter: dict[str, Any] = {
                    "$and": [
                        {"location": {"$eq": location}},
                        {"is_active": {"$eq": is_active}},
                    ]
                }
            else:
                where_filter = {"is_active": {"$eq": is_active}}

            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            relevant_chunks: list[RelevantChunk] = []

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances, strict=False):
                relevant_chunks.append(
                    RelevantChunk(
                        chunk_id=chunk_id,
                        text=doc_text,
                        doc_id=int(meta.get("doc_id", 0)),
                        location=str(meta.get("location", "")),
                        distance=float(dist) if dist is not None else 0.0,
                        metadata=meta,
                    )
                )

            logger.info("Знайдено %d релевантних чанків", len(relevant_chunks))
            return relevant_chunks
        except Exception as exc:
            logger.error("Помилка виконання пошуку в ChromaDB: %s", exc)
            raise VectorStoreError(f"Помилка пошуку у векторному сховищі: {exc}") from exc

    def _deactivate_by_doc_id_sync(self, doc_id: int) -> int:
        """Синхронно деактивує метадані чанків заданого документа."""
        try:
            existing = self._collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=["metadatas"],
            )
            ids = existing.get("ids", [])
            metadatas = existing.get("metadatas", [])

            if not ids or not metadatas:
                return 0

            updated_metadatas = []
            for meta in metadatas:
                new_meta = dict(meta) if meta else {}
                new_meta["is_active"] = False
                updated_metadatas.append(new_meta)

            self._collection.update(ids=ids, metadatas=updated_metadatas)
            logger.info("Деактивовано %d чанків для doc_id=%d", len(ids), doc_id)
            return len(ids)
        except Exception as exc:
            logger.error("Помилка деактивації чанків у ChromaDB: %s", exc)
            raise VectorStoreError(f"Помилка деактивації чанків doc_id={doc_id}: {exc}") from exc

    async def add_chunks(
            self,
            chunks: list[str],
            doc_id: int,
            location: str = "general",
            is_active: bool = True,
            extra_metadatas: list[dict[str, Any]] | None = None,
            embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Асинхронно додає чанки тексту з метаданими у векторну колекцію."""
        return await asyncio.to_thread(
            self._add_chunks_sync,
            chunks,
            doc_id,
            location,
            is_active,
            extra_metadatas,
            embeddings,
        )

    async def search(
            self,
            query: str,
            location: str | None = None,
            top_k: int = 5,
            is_active: bool = True,
            query_embedding: list[float] | None = None,
    ) -> list[RelevantChunk]:
        """Асинхронно шукає релевантні чанки з фільтрацією за локацією та статусом активності."""
        return await asyncio.to_thread(
            self._search_sync,
            query,
            location,
            top_k,
            is_active,
            query_embedding,
        )

    async def deactivate_chunks_by_doc_id(self, doc_id: int) -> int:
        """Асинхронно деактивує (is_active=False) чанки у ChromaDB для старого документа."""
        return await asyncio.to_thread(self._deactivate_by_doc_id_sync, doc_id)
