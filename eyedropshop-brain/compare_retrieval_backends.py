from retrieval import (
    load_resources,
    retrieve_passages,
)
from postgres_retrieval import (
    retrieve_passages_postgres,
)


CASES = [
    (
        "Can I return unopened LUMIFY within 30 days?",
        "en",
    ),
    (
        "What are the ingredients in LUMIFY?",
        "en",
    ),
    (
        "What are the warnings for LUMIFY?",
        "en",
    ),
    (
        "What is your shipping policy?",
        "en",
    ),
    (
        "Do you ship to the United States?",
        "en",
    ),
    (
        "Quelle est votre politique de retour?",
        "fr",
    ),
]


embedding_model, _ = load_resources()

all_passed = True

for index, (
    question,
    language,
) in enumerate(
    CASES,
    start=1,
):
    chroma_passages, _ = (
        retrieve_passages(
            question=question,
            language=language,
            audience="internal",
        )
    )

    postgres_passages, _ = (
        retrieve_passages_postgres(
            question=question,
            language=language,
            audience="internal",
            embedding_model=embedding_model,
        )
    )

    chroma_ids = [
        passage["id"]
        for passage in chroma_passages
    ]

    postgres_ids = [
        passage["id"]
        for passage in postgres_passages
    ]

    passed = (
        chroma_ids
        == postgres_ids
    )

    all_passed = (
        all_passed
        and passed
    )

    print()
    print(
        f"Case {index}: {question}"
    )
    print(
        "Chroma:   ",
        chroma_ids,
    )
    print(
        "Postgres: ",
        postgres_ids,
    )
    print(
        "PASS"
        if passed
        else "FAIL"
    )


print()
print("=" * 60)

if all_passed:
    print(
        "PASS: PostgreSQL retrieval "
        "matches Chroma retrieval."
    )
else:
    print(
        "FAIL: retrieval backends "
        "do not yet match."
    )
