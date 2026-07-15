from __future__ import annotations

import re

from .models import RetrievedChunk
from .evidence import numeric_relation_issues
from .text_processing import normalize_for_lexical


STOP = {
    "oraz", "który", "która", "jest", "przez", "tego", "też", "dla", "się", "nie",
    "może", "należy", "student", "studenta", "regulamin", "studiów", "odpowiedź", "źródła",
    "zgodnie", "przypadku", "został", "została", "swoje", "swojej", "takich", "wszystkich",
}


def _content_terms(text: str) -> set[str]:
    normalized_stop = {normalize_for_lexical(word) for word in STOP}
    words = re.findall(r"[a-z]{4,}", normalize_for_lexical(text))
    return {word[:6] for word in words if word not in normalized_stop}


def grounding_issues(
    answer: str, chunks: list[RetrievedChunk], question: str = ""
) -> list[str]:
    # Metadane cytowania są częścią dozwolonego kontekstu. Bez nich lata w nazwie
    # dokumentu i numery stron były błędnie uznawane za liczby wymyślone przez LLM.
    context = " ".join(f"{chunk.text} {chunk.citation()}" for chunk in chunks)
    issues: list[str] = []
    if len(answer) > 1400:
        issues.append("answer_too_long")

    # Numery stron i lata w nawiasach cytowań są metadanymi, nie twierdzeniami
    # merytorycznymi. Usuwamy je wyłącznie na czas kontroli liczb faktograficznych.
    answer_without_citations = re.sub(r"\[[^\]]+\]", "", answer)
    answer_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", answer_without_citations))
    context_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", context))
    unsupported_numbers = answer_numbers - context_numbers
    if unsupported_numbers:
        issues.append("unsupported_numbers:" + ",".join(sorted(unsupported_numbers)))
    issues.extend(numeric_relation_issues(question, answer, chunks))

    answer_terms = _content_terms(answer)
    if answer_terms:
        overlap = len(answer_terms & _content_terms(context)) / len(answer_terms)
        if overlap < 0.52:
            issues.append(f"low_lexical_grounding:{overlap:.2f}")

    normalized_answer = normalize_for_lexical(answer)
    normalized_context = normalize_for_lexical(context)
    for term in ("formularz", "e-mail", "pocztą", "osobiście", "numer sprawy"):
        normalized_term = normalize_for_lexical(term)
        if normalized_term in normalized_answer and normalized_term not in normalized_context:
            issues.append("unsupported_procedure:" + term)
    return issues


def grounded_fallback(chunks: list[RetrievedChunk]) -> str:
    del chunks
    return (
        "W regulaminie znalazłem powiązane zapisy, ale nie potwierdzają one jednoznacznie "
        "odpowiedzi na to pytanie. Nie będę zgadywać. Doprecyzuj rodzaj decyzji lub "
        "procedury, a sprawdzę właściwy fragment."
    )
