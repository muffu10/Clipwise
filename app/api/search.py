from app.embeddings.embedder import TextEmbedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.consolidator import consolidate_results
from app.rag.generator import AnswerGenerator


class SearchService:
    """
    Clipwise semantic video search and RAG pipeline.

    FAISS -> reranker -> relevance filtering
    -> evidence consolidation -> Gemini
    """

    def __init__(self):
        print("\nLoading Clipwise search models...")

        print("Loading embedding model...")
        self.embedder = TextEmbedder()

        print("\nLoading reranker...")
        self.reranker = Reranker()

        print("\nConnecting to Gemini...")
        self.generator = AnswerGenerator()

        print("\nSearch models ready.")

    def search(
        self,
        video_id: str,
        question: str,
    ) -> dict:
        """
        Search a video's FAISS index and generate
        a grounded answer.
        """

        index_path = (
            f"data/indexes/{video_id}"
        )

        try:
            store = VectorStore.load(
                index_path
            )
        except Exception as error:
            raise FileNotFoundError(
                f"No searchable index found for "
                f"video: {video_id}"
            ) from error

        # ----------------------------------------------
        # 1. Query embedding
        # ----------------------------------------------

        query_embedding = (
            self.embedder.encode_query(
                question
            )
        )

        # ----------------------------------------------
        # 2. FAISS retrieval
        # ----------------------------------------------

        candidates = store.search(
            query_embedding,
            top_k=15,
        )

        if not candidates:
            return {
                "answer": (
                    "The video does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "evidence": [],
            }

        # ----------------------------------------------
        # 3. Cross-encoder reranking
        # ----------------------------------------------

        reranked_results = self.reranker.rerank(
            query=question,
            documents=candidates,
            top_k=5,
        )

        if not reranked_results:
            return {
                "answer": (
                    "The video does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "evidence": [],
            }

        # ----------------------------------------------
        # 4. Relevance filtering
        # ----------------------------------------------

        # Cross-encoder scores are not probabilities.
        # We use a conservative threshold to prevent
        # clearly unrelated questions from reaching Gemini.
        RELEVANCE_THRESHOLD = 0.0

        relevant_results = [
            result
            for result in reranked_results
            if result.get(
                "reranker_score",
                float("-inf"),
            ) >= RELEVANCE_THRESHOLD
        ]

        if not relevant_results:
            return {
                "answer": (
                    "The video does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "evidence": [],
            }

        # ----------------------------------------------
        # 5. Temporal consolidation
        # ----------------------------------------------

        evidence = consolidate_results(
            relevant_results,
            relevance_threshold=RELEVANCE_THRESHOLD,
            top_k=3,
        )

        if not evidence:
            return {
                "answer": (
                    "The video does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "evidence": [],
            }

        # ----------------------------------------------
        # 6. Normalize evidence metadata
        # ----------------------------------------------

        # Ensure every evidence item has the fields
        # expected by the API response.
        normalized_evidence = []

        for item in evidence:
            normalized_item = {
                "start": item.get(
                    "start",
                    0,
                ),
                "end": item.get(
                    "end",
                    0,
                ),
                "text": item.get(
                    "text",
                    "",
                ),
                "reranker_score": item.get(
                    "reranker_score",
                    0.0,
                ),
                "chunk_count": item.get(
                    "chunk_count",
                    1,
                ),
            }

            normalized_evidence.append(
                normalized_item
            )

        if not normalized_evidence:
            return {
                "answer": (
                    "The video does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "evidence": [],
            }

        # ----------------------------------------------
        # 7. Gemini RAG generation
        # ----------------------------------------------

        answer = self.generator.generate(
            question=question,
            evidence=normalized_evidence,
        )

        return {
            "answer": answer,
            "evidence": normalized_evidence,
        }