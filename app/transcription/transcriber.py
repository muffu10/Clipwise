from pathlib import Path
import json

from faster_whisper import WhisperModel


class VideoTranscriber:
    """
    Transcribe videos using faster-whisper.

    CPU configuration:
    - int8 quantization
    - CPU inference
    - Timestamped segments
    """

    def __init__(
        self,
        model_name: str = "base",
    ):
        self.model_name = model_name

        print(
            f"Loading faster-whisper model: "
            f"{model_name}"
        )

        print(
            "Using device: cpu"
        )

        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

        print(
            "faster-whisper model ready."
        )

    def transcribe(
        self,
        video_path: str,
    ) -> dict:
        """
        Transcribe a video and return
        timestamped segments.
        """

        video = Path(video_path)

        if not video.exists():
            raise FileNotFoundError(
                f"Video not found: {video}"
            )

        print(
            f"\nTranscribing: {video.name}"
        )

        print(
            "Using faster-whisper CPU INT8..."
        )

        segments, info = self.model.transcribe(
            str(video),
            beam_size=1,
            vad_filter=True,
        )

        transcript_segments = []
        full_text = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            transcript_segments.append(
                {
                    "start": round(
                        segment.start,
                        2,
                    ),
                    "end": round(
                        segment.end,
                        2,
                    ),
                    "text": text,
                }
            )

            full_text.append(text)

        return {
            "video": video.name,
            "language": info.language,
            "text": " ".join(
                full_text
            ),
            "segments": transcript_segments,
        }

    def save_transcript(
        self,
        transcript: dict,
        output_path: str,
    ) -> None:
        """
        Save transcript data as JSON.
        """

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                transcript,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"\nTranscript saved to: "
            f"{output}"
        )


def transcribe_video(
    video_path: str,
    output_path: str,
    model_name: str = "base",
):
    """
    Convenience function for
    transcribing a video.
    """

    transcriber = VideoTranscriber(
        model_name
    )

    transcript = transcriber.transcribe(
        video_path
    )

    transcriber.save_transcript(
        transcript,
        output_path,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a video with "
            "faster-whisper."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the video file",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/transcripts/"
            "transcript.json"
        ),
        help="Output JSON path",
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

    transcribe_video(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model,
    )