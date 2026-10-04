from app.transcription.transcriber import (
    VideoTranscriber,
)
from app.embeddings.embedder import (
    TextEmbedder,
)
from app.retrieval.reranker import (
    Reranker,
)
from app.rag.generator import (
    AnswerGenerator,
)


class ModelManager:
    """
    Loads and keeps Clipwise AI models in memory.

    Models are loaded once when the application starts
    instead of being loaded for every video/search request.
    """

    def __init__(
        self,
        whisper_model: str = "base",
    ):
        print("\n" + "=" * 60)
        print("LOADING CLIPWISE AI MODELS")
        print("=" * 60)

        print("\n[1/4] Loading Whisper...")
        self.transcriber = VideoTranscriber(
            model_name=whisper_model
        )

        print("\n[2/4] Loading embedding model...")
        self.embedder = TextEmbedder()

        print("\n[3/4] Loading reranker...")
        self.reranker = Reranker()

        print("\n[4/4] Connecting to Gemini...")
        self.generator = AnswerGenerator()

        print("\n" + "=" * 60)
        print("ALL CLIPWISE MODELS READY")
        print("=" * 60 + "\n")