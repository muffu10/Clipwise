import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


REGISTRY_PATH = (
    Path("data")
    / "videos"
    / "registry.json"
)


def _load_registry() -> list[dict]:
    """Load registered videos."""

    if not REGISTRY_PATH.exists():
        return []

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _save_registry(
    videos: list[dict],
) -> None:
    """Save registered videos."""

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REGISTRY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            videos,
            file,
            indent=4,
            ensure_ascii=False,
        )


def create_video_record(
    filename: str,
) -> dict:
    """Create and register a new video."""

    videos = _load_registry()

    video_id = str(uuid4())

    record = {
        "video_id": video_id,
        "filename": filename,
        "status": "uploaded",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    videos.append(record)

    _save_registry(videos)

    return record


def update_video_status(
    video_id: str,
    status: str,
    **extra,
) -> dict | None:
    """Update the status of a registered video."""

    videos = _load_registry()

    for video in videos:

        if video["video_id"] == video_id:

            video["status"] = status

            video.update(extra)

            _save_registry(videos)

            return video

    return None


def get_video(
    video_id: str,
) -> dict | None:
    """Retrieve a video record."""

    videos = _load_registry()

    for video in videos:

        if video["video_id"] == video_id:
            return video

    return None