from pathlib import Path

from app.transcription.transcriber import VideoTranscriber
from app.chunking.chunker import create_chunks
from app.embeddings.embedder import TextEmbedder
from app.retrieval.vector_store import VectorStore


class VideoProcessor:
    """
    Complete Clipwise video ingestion pipeline.

    Video
        -> Whisper transcription
        -> Semantic chunking
        -> Embeddings
        -> FAISS index
    """

    def __init__(
        self,
        whisper_model: str = "base",
    ):
        self.whisper_model = whisper_model

    def process(
        self,
        video_path: str,
        video_id: str,
    ) -> dict:
        """
        Process a video and create all searchable artifacts.
        """

        video = Path(video_path)

        if not video.exists():
            raise FileNotFoundError(
                f"Video not found: {video}"
            )

        # --------------------------------------------------
        # 1. Transcription
        # --------------------------------------------------

        print("\n[1/4] Transcribing video...")

        transcriber = VideoTranscriber(
            model_name=self.whisper_model
        )

        transcript = transcriber.transcribe(
            str(video)
        )

        transcript_path = (
            Path("data")
            / "transcripts"
            / f"{video_id}.json"
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

        for chunk in chunks:
            chunk["video"] = video.name
            chunk["video_id"] = video_id

        print(
            f"Created {len(chunks)} chunks."
        )

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
            / video_id
        )

        store.save(
            str(index_path)
        )

        return {
            "video_id": video_id,
            "filename": video.name,
            "transcript_path": str(
                transcript_path
            ),
            "index_path": str(
                index_path
            ),
            "chunk_count": len(chunks),
            "duration": (
                transcript["segments"][-1]["end"]
                if transcript["segments"]
                else 0
            ),
        }