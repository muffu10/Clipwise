from app.rag.generator import AnswerGenerator


def main():

    evidence = [
        {
            "start": 23.0,
            "end": 69.0,
            "text": (
                "There are two different ways that "
                "you can get started with Excel. "
                "The first way, you can navigate to "
                "excel.new in your web browser. "
                "You'll need to log in, but that's "
                "entirely free. Excel on the web has "
                "most of the functionality that "
                "you'll find in the desktop app. "
                "Second, you can also install Excel "
                "on your desktop, but you will need "
                "to purchase Microsoft 365."
            ),
            "reranker_score": 8.34,
            "chunk_count": 2,
        }
    ]

    question = "How can I get Excel?"

    generator = AnswerGenerator()

    answer = generator.generate(
        question=question,
        evidence=evidence,
    )

    print("\nClipwise Answer")
    print("=" * 40)
    print(answer)


if __name__ == "__main__":
    main()