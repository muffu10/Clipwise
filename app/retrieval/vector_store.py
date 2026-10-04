from pathlib import Path
import json

import faiss
import numpy as np


class VectorStore:
    """FAISS vector database for Clipwise."""

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner Product works as cosine similarity because
        # our embeddings are normalized.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.documents = []

    def add(
        self,
        embeddings,
        documents,
    ):
        """Add embeddings and their metadata."""

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(vectors)

        self.documents.extend(
            documents
        )

    def search(
        self,
        query_embedding,
        top_k: int = 15,
    ):
        """
        Retrieve the most semantically similar
        candidates using FAISS.

        This method intentionally does not perform
        final relevance filtering or deduplication.

        FAISS is only responsible for candidate retrieval.
        """

        if not self.documents:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        candidate_count = min(
            top_k,
            len(self.documents),
        )

        scores, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        candidates = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            document = self.documents[index].copy()

            document["faiss_score"] = float(
                score
            )

            candidates.append(
                document
            )

        return candidates

    def save(self, directory: str):
        """Save the FAISS index and metadata."""

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path / "index.faiss"),
        )

        with open(
            path / "documents.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, directory: str):
        """Load an existing vector store."""

        path = Path(directory)

        index = faiss.read_index(
            str(path / "index.faiss")
        )

        with open(
            path / "documents.json",
            "r",
            encoding="utf-8",
        ) as file:

            documents = json.load(file)

        store = cls(index.d)

        store.index = index
        store.documents = documents

        return store