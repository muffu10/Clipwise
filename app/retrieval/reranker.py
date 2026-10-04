from typing import List, Dict

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks retrieved transcript chunks using
    a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-rank candidate documents according
        to their relevance to the query.
        """

        if not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for document, score in zip(
            documents,
            scores,
        ):
            result = document.copy()

            result["reranker_score"] = float(
                score
            )

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]