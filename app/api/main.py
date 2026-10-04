from pathlib import Path
import shutil

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.video_processor import (
    VideoProcessor,
)
from app.services.video_registry import (
    create_video_record,
    get_video,
    update_video_status,
    _load_registry,
)
from app.api.search import SearchService


app = FastAPI(
    title="Clipwise API",
    description=(
        "AI-powered video understanding "
        "and semantic search API."
    ),
    version="0.1.0",
)


# Allow the React frontend to communicate
# with the FastAPI backend during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VIDEO_DIRECTORY = (
    Path("data")
    / "videos"
)


search_service = SearchService()


class SearchRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "name": "Clipwise API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/videos")
def upload_video(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A video file is required.",
        )

    allowed_extensions = {
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Supported formats: "
                "mp4, webm, mov, mkv, avi."
            ),
        )

    record = create_video_record(
        file.filename
    )

    video_id = record["video_id"]

    VIDEO_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        VIDEO_DIRECTORY
        / f"{video_id}{extension}"
    )

    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        update_video_status(
            video_id,
            "processing",
        )

        processor = VideoProcessor(
            whisper_model="base"
        )

        result = processor.process(
            video_path=str(video_path),
            video_id=video_id,
        )

        update_video_status(
            video_id,
            "ready",
            chunk_count=result[
                "chunk_count"
            ],
            duration=result[
                "duration"
            ],
        )

        return {
            "video_id": video_id,
            "filename": file.filename,
            "status": "ready",
            "chunk_count": result[
                "chunk_count"
            ],
            "duration": result[
                "duration"
            ],
        }

    except Exception as error:
        update_video_status(
            video_id,
            "failed",
            error=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/videos")
def list_videos():
    videos = _load_registry()

    videos.sort(
        key=lambda video: video.get(
            "created_at",
            "",
        ),
        reverse=True,
    )

    return videos


@app.get("/videos/{video_id}/file")
def get_video_file(
    video_id: str,
):
    video = get_video(
        video_id
    )

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    matches = list(
        VIDEO_DIRECTORY.glob(
            f"{video_id}.*"
        )
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Video file not found.",
        )

    return FileResponse(
        path=matches[0]
    )


@app.get("/videos/{video_id}")
def video_status(
    video_id: str,
):
    video = get_video(
        video_id
    )

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    return video


@app.post("/videos/{video_id}/search")
def search_video(
    video_id: str,
    request: SearchRequest,
):
    video = get_video(
        video_id
    )

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    if video["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video is not ready. "
                f"Current status: "
                f"{video['status']}"
            ),
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = search_service.search(
            video_id=video_id,
            question=request.question,
        )

        return {
            "video_id": video_id,
            "question": request.question,
            "answer": result["answer"],
            "evidence": [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "relevance_score": item[
                        "reranker_score"
                    ],
                    "chunk_count": item[
                        "chunk_count"
                    ],
                }
                for item in result["evidence"]
            ],
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )