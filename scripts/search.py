import argparse

from app.embeddings.embedder import TextEmbedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.consolidator import consolidate_results
from app.rag.generator import AnswerGenerator


def format_timestamp(seconds: float) -> str:
    """Convert seconds into HH:MM:SS or MM:SS."""

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Search a Clipwise video and generate grounded answers."
    )

    parser.add_argument(
        "index",
        help="Name of the vector index",
    )

    args = parser.parse_args()

    index_path = (
        f"data/indexes/{args.index}"
    )

    print("Loading embedding model...")

    embedder = TextEmbedder()

    print("\nLoading reranker...")

    reranker = Reranker()

    print("\nConnecting to Gemini...")

    generator = AnswerGenerator()

    print(
        f"\nLoading index: {index_path}"
    )

    store = VectorStore.load(
        index_path
    )

    print("\nClipwise AI Video Search")
    print("=" * 40)

    while True:

        query = input(
            "\nAsk something about the video "
            "(or type 'exit'): "
        )

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        # -------------------------------------------------
        # 1. Convert question into an embedding
        # -------------------------------------------------

        query_embedding = (
            embedder.encode_query(query)
        )

        # -------------------------------------------------
        # 2. Retrieve semantic candidates using FAISS
        # -------------------------------------------------

        candidates = store.search(
            query_embedding,
            top_k=15,
        )

        # -------------------------------------------------
        # 3. Rerank candidates using cross-encoder
        # -------------------------------------------------

        reranked_results = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=5,
        )

        # -------------------------------------------------
        # 4. Consolidate relevant evidence
        # -------------------------------------------------

        results = consolidate_results(
            reranked_results,
            relevance_threshold=0.0,
            top_k=3,
        )

        print("\nGenerating answer...")

        # -------------------------------------------------
        # 5. Generate grounded answer using Gemma
        # -------------------------------------------------

        answer = generator.generate(
            question=query,
            evidence=results,
        )

        print("\nClipwise Answer")
        print("=" * 40)
        print(answer)

        # -------------------------------------------------
        # 6. Display retrieved evidence
        # -------------------------------------------------

        print("\nSupporting video evidence")
        print("-" * 40)

        for result in results:

            start = format_timestamp(
                result["start"]
            )

            end = format_timestamp(
                result["end"]
            )

            print(
                f"\n[{start} - {end}]"
            )

            print(
                f"Relevance score: "
                f"{result['reranker_score']:.4f}"
            )

            print(
                f"Combined chunks: "
                f"{result['chunk_count']}"
            )


if __name__ == "__main__":
    main()