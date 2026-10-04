from typing import List

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Generate semantic embeddings for text."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: List[str],
    ):
        """Convert text into embedding vectors."""

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, query: str):
        """Generate an embedding for a user query."""

        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]