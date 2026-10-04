import argparse
from pathlib import Path

from app.transcription.transcriber import VideoTranscriber
from app.chunking.chunker import create_chunks
from app.embeddings.embedder import TextEmbedder
from app.retrieval.vector_store import VectorStore


def process_video(
    video_path: str,
    model_name: str = "base",
):
    """
    Process a video from raw file to searchable FAISS index.

    Pipeline:
        Video
        -> Whisper transcript
        -> Semantic chunks
        -> Embeddings
        -> FAISS index
    """

    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}"
        )

    print("=" * 60)
    print("CLIPWISE VIDEO PROCESSING")
    print("=" * 60)

    print(f"\nVideo: {video.name}")

    # --------------------------------------------------
    # 1. Transcription
    # --------------------------------------------------

    print("\n[1/4] Transcribing video...")

    transcriber = VideoTranscriber(
        model_name=model_name
    )

    transcript = transcriber.transcribe(
        str(video)
    )

    transcript_path = (
        Path("data")
        / "transcripts"
        / f"{video.stem}.json"
    )

    transcriber.save_transcript(
        transcript,
        str(transcript_path),
    )

    # --------------------------------------------------
    # 2. Semantic chunking
    # --------------------------------------------------

    print("\n[2/4] Creating semantic chunks...")

    chunks = create_chunks(
        transcript["segments"],
        max_words=80,
        overlap_segments=1,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Attach video metadata.
    for chunk in chunks:
        chunk["video"] = video.name

    # --------------------------------------------------
    # 3. Embeddings
    # --------------------------------------------------

    print("\n[3/4] Generating embeddings...")

    embedder = TextEmbedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(
        texts
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    # --------------------------------------------------
    # 4. FAISS index
    # --------------------------------------------------

    print("\n[4/4] Building FAISS index...")

    store = VectorStore(
        dimension=embeddings.shape[1]
    )

    store.add(
        embeddings,
        chunks,
    )

    index_path = (
        Path("data")
        / "indexes"
        / video.stem
    )

    store.save(
        str(index_path)
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"\nTranscript: {transcript_path}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"FAISS index: {index_path}"
    )

    print(
        "\nThe video is now ready for semantic search."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a video through the complete "
            "Clipwise ingestion pipeline."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the video file",
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
        ],
        help="Whisper model size",
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()