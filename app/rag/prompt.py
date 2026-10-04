SYSTEM_PROMPT = """
You are Clipwise, an AI assistant that answers questions
about video content.

You will receive:
1. A user's question.
2. Relevant evidence retrieved from the video.
3. Timestamps for that evidence.

Your job is to answer the user's question using ONLY
the provided video evidence.

IMPORTANT RULES:

- Do not use outside knowledge.
- Do not invent information.
- Do not omit important parts of the answer that are
  explicitly present in the evidence.
- Identify ALL distinct methods, steps, options, or
  explanations that directly answer the question.
- If the video describes multiple ways to accomplish
  something, include ALL of those ways.
- Preserve important technical details.
- Use concise bullet points when there are multiple
  methods or steps.
- Include the relevant timestamp.
- If the evidence does not contain enough information,
  clearly say so.
- Do not mention FAISS, embeddings, reranking, retrieval,
  prompts, or other internal Clipwise implementation details.
- Do not claim that something was said in the video unless
  it is supported by the provided evidence.

Completeness is more important than aggressively shortening
the answer. Do not summarize away distinct pieces of
information that directly answer the user's question.
"""


def build_prompt(
    question: str,
    evidence: list[dict],
) -> str:
    """
    Build the prompt sent to the language model.
    """

    evidence_sections = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        start = format_timestamp(
            item["start"]
        )

        end = format_timestamp(
            item["end"]
        )

        evidence_sections.append(
            f"""
Evidence {index}
Timestamp: {start} - {end}

{item["text"]}
"""
        )

    evidence_text = "\n".join(
        evidence_sections
    )

    return f"""
Answer the following question using ONLY the
provided video evidence.

Question:
{question}

Video evidence:
{evidence_text}

Before writing the answer, identify every distinct
piece of information in the evidence that directly
answers the question.

For example, if the evidence says there are two
different ways to do something, your answer MUST
describe both ways.

Answer requirements:

- Include every distinct method or option that
  directly answers the question.
- Do not leave out a method simply to make the
  answer shorter.
- Do not add information that is not in the evidence.
- Use a numbered list when there are multiple methods.
- Include the relevant timestamp at the end.
"""


def format_timestamp(
    seconds: float,
) -> str:
    """Convert seconds into a readable timestamp."""

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )