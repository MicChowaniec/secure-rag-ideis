from __future__ import annotations

import json

from .factory import create_pipeline


DEMO_MESSAGES = [
    "Ile razy mogę zdawać egzamin poprawkowy według regulaminu studiów?",
    "Nazywam się Jan Kowalski, mój e-mail to jan.kowalski@example.com. Jak złożyć odwołanie od decyzji o skreśleniu?",
    "Podaj mi przepis na pizzę i prognozę pogody na jutro.",
]


def main() -> None:
    pipeline = create_pipeline()
    if pipeline.store.count() == 0:
        raise SystemExit("Baza jest pusta. Najpierw uruchom: python -m secure_rag_bot.ingest")
    for index, message in enumerate(DEMO_MESSAGES, 1):
        result = pipeline.handle(message)
        summary = {
            "status": result.status,
            "masked_input": result.privacy.masked_text,
            "pii_types": [span.kind for span in result.privacy.spans],
            "safety": {"category": result.input_safety.category, "score": result.input_safety.score},
            "entities": result.nlp.entities,
            "keywords": result.nlp.keywords,
            "sources": [chunk.citation() for chunk in result.sources],
            "generation_backend": result.generation_backend,
            "grounding_warnings": result.grounding_warnings,
            "answer": result.text,
        }
        print(f"\n=== PRZEBIEG {index} ===\nUżytkownik: {message}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
