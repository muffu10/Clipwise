import os

from dotenv import load_dotenv
from google import genai

from app.rag.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
)


class AnswerGenerator:
    """
    Generate concise, grounded answers using
    Google's Gemini cloud model.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
    ):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in the .env file."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        question: str,
        evidence: list[dict],
    ) -> str:
        """
        Generate an answer grounded only
        in the retrieved video evidence.
        """

        if not evidence:
            return (
                "The video does not contain enough "
                "information to answer this question."
            )

        prompt = build_prompt(
            question=question,
            evidence=evidence,
        )

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "thinking_level": "minimal",
                "max_output_tokens": 500,
            },
        )

        return interaction.output_text.strip()