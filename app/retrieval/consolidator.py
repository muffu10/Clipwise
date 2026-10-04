from typing import List, Dict


def consolidate_results(
    results: List[Dict],
    relevance_threshold: float = 0.0,
    top_k: int = 3,
    continuation_gap: float = 15.0,
) -> List[Dict]:
    """
    Convert reranked chunks into coherent evidence blocks.

    Strong results are selected first. Nearby chunks that appear
    to be continuations of a strong result may also be included,
    even when their reranker score is lower.
    """

    if not results:
        return []

    ranked = sorted(
        results,
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    # Strong candidates.
    strong_candidates = [
        item
        for item in ranked
        if item["reranker_score"] >= relevance_threshold
    ]

    if not strong_candidates:
        return [ranked[0]]

    selected = []

    # Always keep the strongest result.
    best = strong_candidates[0]

    selected.append(best)

    # Find nearby continuation chunks.
    for candidate in ranked[1:]:

        if candidate in selected:
            continue

        # Don't automatically include distant weak results.
        if candidate["reranker_score"] < relevance_threshold:

            if not _is_nearby_continuation(
                candidate,
                best,
                continuation_gap,
            ):
                continue

        # Avoid near-duplicate text.
        duplicate = False

        for existing in selected:

            if _text_overlap(
                candidate["text"],
                existing["text"],
            ) >= 0.55:
                duplicate = True
                break

        if duplicate:
            continue

        # A candidate can be included if it is
        # temporally connected to existing evidence.
        connected = False

        for existing in selected:

            if _is_connected(
                candidate,
                existing,
            ):
                connected = True
                break

        if connected:
            selected.append(candidate)

    # Build temporal evidence blocks.
    blocks = []

    for candidate in selected:

        merged = False

        for block in blocks:

            if _is_connected(
                candidate,
                block,
            ):
                _merge_into_block(
                    block,
                    candidate,
                )

                merged = True
                break

        if not merged:

            blocks.append(
                {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "text": candidate["text"],
                    "reranker_score": candidate[
                        "reranker_score"
                    ],
                    "chunk_count": 1,
                }
            )

    # Strongest blocks first.
    blocks.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    blocks = blocks[:top_k]

    # Chronological presentation.
    blocks.sort(
        key=lambda item: item["start"],
    )

    return blocks


def _is_nearby_continuation(
    candidate: Dict,
    reference: Dict,
    max_gap: float,
) -> bool:
    """
    Determine whether a lower-scoring candidate is
    temporally close enough to be a continuation.
    """

    gap = _temporal_gap(
        candidate,
        reference,
    )

    if gap <= max_gap:
        return True

    return _time_overlap(
        candidate,
        reference,
    ) > 0


def _is_connected(
    candidate: Dict,
    reference: Dict,
) -> bool:
    """
    Determine whether two chunks belong to the
    same temporal evidence block.
    """

    time_overlap = _time_overlap(
        candidate,
        reference,
    )

    gap = _temporal_gap(
        candidate,
        reference,
    )

    text_overlap = _text_overlap(
        candidate["text"],
        reference["text"],
    )

    # Overlapping chunks with some shared vocabulary.
    if (
        time_overlap >= 0.20
        and text_overlap >= 0.05
    ):
        return True

    # Adjacent chunks with some shared vocabulary.
    if (
        gap <= 2.0
        and text_overlap >= 0.05
    ):
        return True

    return False


def _merge_into_block(
    block: Dict,
    candidate: Dict,
) -> None:
    """Merge a candidate into an evidence block."""

    block["start"] = min(
        block["start"],
        candidate["start"],
    )

    block["end"] = max(
        block["end"],
        candidate["end"],
    )

    if candidate["text"] not in block["text"]:
        block["text"] += " " + candidate["text"]

    block["reranker_score"] = max(
        block["reranker_score"],
        candidate["reranker_score"],
    )

    block["chunk_count"] += 1


def _text_overlap(
    text_a: str,
    text_b: str,
) -> float:
    """Calculate word-level Jaccard overlap."""

    words_a = set(
        text_a.lower().split()
    )

    words_b = set(
        text_b.lower().split()
    )

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def _time_overlap(
    candidate: Dict,
    reference: Dict,
) -> float:
    """Calculate temporal overlap ratio."""

    overlap_start = max(
        candidate["start"],
        reference["start"],
    )

    overlap_end = min(
        candidate["end"],
        reference["end"],
    )

    if overlap_end <= overlap_start:
        return 0.0

    overlap_duration = (
        overlap_end - overlap_start
    )

    candidate_duration = (
        candidate["end"]
        - candidate["start"]
    )

    if candidate_duration <= 0:
        return 0.0

    return (
        overlap_duration
        / candidate_duration
    )


def _temporal_gap(
    candidate: Dict,
    reference: Dict,
) -> float:
    """Calculate the gap between two time ranges."""

    if candidate["start"] > reference["end"]:
        return (
            candidate["start"]
            - reference["end"]
        )

    if reference["start"] > candidate["end"]:
        return (
            reference["start"]
            - candidate["end"]
        )

    return 0.0