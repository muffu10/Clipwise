from typing import List, Dict


def create_chunks(
    segments: List[Dict],
    max_words: int = 80,
    overlap_segments: int = 1,
) -> List[Dict]:
    """
    Combine Whisper segments into larger semantic chunks.

    Each chunk keeps:
    - start timestamp
    - end timestamp
    - text
    - segment count
    """

    if not segments:
        return []

    chunks = []
    current_segments = []
    current_word_count = 0

    for segment in segments:
        text = segment["text"].strip()

        if not text:
            continue

        word_count = len(text.split())

        # If adding this segment would make the chunk too large,
        # save the current chunk first.
        if (
            current_segments
            and current_word_count + word_count > max_words
        ):
            chunks.append(_build_chunk(current_segments))

            # Keep a small overlap between chunks.
            if overlap_segments > 0:
                current_segments = current_segments[
                    -overlap_segments:
                ]
                current_word_count = sum(
                    len(item["text"].split())
                    for item in current_segments
                )
            else:
                current_segments = []
                current_word_count = 0

        current_segments.append(segment)
        current_word_count += word_count

    # Save the final chunk.
    if current_segments:
        chunks.append(_build_chunk(current_segments))

    # Add stable IDs.
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = index

    return chunks


def _build_chunk(segments: List[Dict]) -> Dict:
    """Build one chunk from multiple Whisper segments."""

    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(
            segment["text"].strip()
            for segment in segments
        ),
        "segment_count": len(segments),
    }