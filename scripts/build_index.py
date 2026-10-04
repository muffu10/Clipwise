import argparse
import json
from pathlib import Path

from app.chunking.chunker import create_chunks
from app.embeddings.embedder import TextEmbedder
from app.retrieval.vector_store import VectorStore


def build_index(
    transcript_path: str,
    index_name: str,
):
    transcript_file = Path(transcript_path)

    if not transcript_file.exists():
        raise FileNotFoundError(
            f"Transcript not found: {transcript_file}"
        )

    print("Loading transcript...")

    with transcript_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        transcript = json.load(file)

    print(
        f"Video: {transcript['video']}"
    )

    print("\nCreating semantic chunks...")

    chunks = create_chunks(
        transcript["segments"],
        max_words=80,
        overlap_segments=1,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\nGenerating embeddings...")

    embedder = TextEmbedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts)

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print("\nBuilding FAISS index...")

    store = VectorStore(
        dimension=embeddings.shape[1]
    )

    # Attach video information to every chunk.
    for chunk in chunks:
        chunk["video"] = transcript["video"]

    store.add(
        embeddings,
        chunks,
    )

    index_path = (
        Path("data")
        / "indexes"
        / index_name
    )

    store.save(
        str(index_path)
    )

    print(
        f"\nIndex saved to: {index_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a semantic FAISS index from a Clipwise transcript."
    )

    parser.add_argument(
        "transcript",
        help="Path to transcript JSON",
    )

    parser.add_argument(
        "--name",
        help="Name of the vector index",
    )

    args = parser.parse_args()

    transcript_path = Path(
        args.transcript
    )

    index_name = args.name

    if not index_name:
        index_name = transcript_path.stem

    build_index(
        transcript_path=str(transcript_path),
        index_name=index_name,
    )


if __name__ == "__main__":
    main()