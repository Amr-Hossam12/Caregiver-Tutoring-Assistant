"""
ChromaDB-backed RAG store for MathDial misconceptions.

At startup, documents from train.csv are embedded and persisted.  Subsequent
runs reuse the persisted collection so re-indexing is skipped.
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List, Dict, Any

COLLECTION_NAME = "mathdial_misconceptions"
_BATCH_SIZE = 100


class MisconceptionRAG:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._ef = DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, docs: List[Dict[str, Any]], force: bool = False) -> None:
        """Embed and store misconception documents.  Skips if already indexed."""
        if not force and self.collection.count() > 0:
            print(f"[RAG] Collection already has {self.collection.count()} docs — skipping re-index.")
            return

        if force:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._ef,
            )

        texts, metadatas, ids = [], [], []
        for i, doc in enumerate(docs):
            # Text to embed: misconception label + truncated student attempt
            texts.append(
                f"Misconception: {doc['misconception']}\n"
                f"Student attempt: {doc['student_attempt'][:400]}"
            )
            metadatas.append({
                "qid": doc["qid"],
                "misconception": doc["misconception"],
                "example_teacher_response": doc["example_teacher_response"][:500],
                "example_move_type": doc["example_move_type"],
                "self_correctness": doc["self_correctness"],
                "problem": doc["problem"][:300],
            })
            ids.append(f"doc_{i}")

        for start in range(0, len(texts), _BATCH_SIZE):
            self.collection.add(
                documents=texts[start : start + _BATCH_SIZE],
                metadatas=metadatas[start : start + _BATCH_SIZE],
                ids=ids[start : start + _BATCH_SIZE],
            )
        print(f"[RAG] Indexed {len(texts)} documents into '{COLLECTION_NAME}'.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        student_attempt: str,
        problem: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the top-n most similar misconception examples."""
        query_text = (
            f"Student attempt: {student_attempt[:400]}\n"
            f"Problem context: {problem[:300]}"
        )
        n = min(n_results, self.collection.count())
        if n == 0:
            return []

        results = self.collection.query(query_texts=[query_text], n_results=n)

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "misconception": results["metadatas"][0][i]["misconception"],
                "example_teacher_response": results["metadatas"][0][i]["example_teacher_response"],
                "example_move_type": results["metadatas"][0][i]["example_move_type"],
                "self_correctness": results["metadatas"][0][i]["self_correctness"],
                "distance": results["distances"][0][i],
            })
        return retrieved
