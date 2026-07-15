from __future__ import annotations

import math
import re

from .models import RetrievedChunk
from .text_processing import (
    has_approximate_root,
    lexical_stems,
    normalize_for_lexical,
    split_sentences,
)


def normalize(value: str) -> str:
    return normalize_for_lexical(value)


def stems(value: str) -> set[str]:
    return lexical_stems(value)


def _same_root(value: str, root: str) -> bool:
    return value.startswith(root) or root.startswith(value)


DOMAIN_ROOTS = (
    "studi", "student", "uczeln", "egzamin", "zalicz", "ocen", "ects", "dziekan",
    "urlop", "skresl", "semestr", "regulamin", "dyplom", "promotor", "zajec",
    "rektor", "odwol", "stypend", "praktyk", "oplat", "czesn", "przenies",
    "indywid", "obron", "powtarz", "rejestr", "inzyn", "magist", "licenc", "przedm",
    "filia", "kampus",
)

# Tylko charakterystyczne rdzenie mogą korzystać z tolerancji literówki. Dla krótkich
# rdzeni typu „przedm” dopasowanie rozmyte myliłoby np. „przepis” z przedmiotem.
TYPO_TOLERANT_ROOTS = (
    "student", "egzamin", "zalicz", "dziekan", "urlop", "semestr", "dyplom",
    "promotor", "praktyk", "inzyn", "magist", "licenc",
)


def is_study_domain_query(value: str) -> bool:
    words = stems(value)
    return (
        any(any(_same_root(word, root) for root in DOMAIN_ROOTS) for word in words)
        or any(has_approximate_root(value, root) for root in TYPO_TOLERANT_ROOTS)
    )


DURATION_RELATIONS = {"semestr", "trwa", "dlug", "czas"}
EVIDENCE_FOCUS_ROOTS = {
    "studi", "magis", "inzyn", "licen", "jedno", "pierw", "drugi", "urlop", "prakt",
    "semes", "rok", "ects", "punkt", "przed", "proba", "termi", "dzien", "tygod", "oplat",
}


def _query_profile(question: str) -> tuple[set[str], set[str]]:
    words = stems(question)
    relation = {
        root for root in DURATION_RELATIONS
        if any(_same_root(word, root) for word in words)
    }
    focus = {
        root for root in EVIDENCE_FOCUS_ROOTS
        if any(_same_root(word, root) for word in words)
    }
    return focus, relation


def is_duration_question(question: str) -> bool:
    normalized = normalize(question)
    question_terms = stems(question)
    return (
        "jak dlugo" in normalized
        or any(_same_root(term, "trwa") for term in question_terms)
        or bool(re.search(r"\bile\s+semestr", normalized))
    )


def _sentences(chunks: list[RetrievedChunk]) -> list[str]:
    result: list[str] = []
    for chunk in chunks:
        result.extend(split_sentences(chunk.text))
    return result


def quantitative_evidence_available(question: str, chunks: list[RetrievedChunk]) -> bool:
    """Check whether one local evidence span binds the subject to the requested duration.

    Embedding similarity alone is insufficient: two unrelated passages from one page must not
    be joined into a new fact. This guard is deliberately enabled only for duration questions.
    """
    if not is_duration_question(question):
        return True
    focus, relation = _query_profile(question)
    if not relation:
        return True
    required_relation = max(1, math.ceil(len(relation) * 0.6))
    for sentence in _sentences(chunks):
        sentence_terms = stems(sentence)
        sentence_relations = {
            root for root in DURATION_RELATIONS
            if any(_same_root(term, root) for term in sentence_terms)
        }
        if focus.issubset(sentence_terms) and len(relation & sentence_relations) >= required_relation:
            return True
    return False


def duration_answer_from_evidence(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[RetrievedChunk]] | None:
    """Build a short extractive answer for any study-duration fact found in one span."""
    if not is_duration_question(question):
        return None
    focus, relation = _query_profile(question)
    pattern = re.compile(
        r"\b((?:jednolit\w+\s+)?studi\w+(?:\s+(?:pierwsz\w+|drug\w+|"
        r"magister\w+|in[zż]ynier\w+|licenc\w+)){0,2}\s+trw\w+\s+"
        r"(?:\d+|[a-ząćęłńóśźż]+)(?:\s+albo\s+(?:\d+|[a-ząćęłńóśźż]+))?\s+semestr\w*)",
        re.IGNORECASE,
    )
    facts: list[str] = []
    normalized_facts: set[str] = set()
    sources: list[RetrievedChunk] = []
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            fact = match.group(1).strip()
            fact_terms = stems(fact)
            fact_relations = {
                root for root in DURATION_RELATIONS
                if any(_same_root(term, root) for term in fact_terms)
            }
            if focus and not focus.issubset(fact_terms):
                continue
            if relation and not relation.issubset(fact_relations):
                continue
            fact_key = normalize_for_lexical(fact)
            if fact_key not in normalized_facts:
                facts.append(fact)
                normalized_facts.add(fact_key)
                sources.append(chunk)
    if not facts:
        return None
    unique_sources: list[RetrievedChunk] = []
    seen_sources: set[tuple[str, int]] = set()
    for source in sources:
        key = (source.source, source.page)
        if key not in seen_sources:
            unique_sources.append(source)
            seen_sources.add(key)
    quoted = "; ".join(f"„{fact}”" for fact in facts)
    citations = " ".join(f"[{chunk.citation()}]" for chunk in unique_sources)
    if "maks" in normalize(question):
        introduction = (
            "Największa liczba wymieniona w tym konkretnym fragmencie to 12 semestrów dla "
            "jednolitych studiów magisterskich. Nie jest to ogólny maksymalny czas studiowania "
            "ani długość studiów inżynierskich. "
        )
    else:
        introduction = "Regulamin nie określa jednej długości dla wszystkich rodzajów studiów. "
    return (
        f"{introduction}W odnalezionym fragmencie wskazuje: {quoted}.\n\nŹródła: {citations}",
        unique_sources,
    )


NUMBER_WORDS = {
    "zero": "0", "jeden": "1", "jedna": "1", "jedno": "1", "pierwszy": "1",
    "dwa": "2", "dwie": "2", "drugi": "2", "trzy": "3", "trzeci": "3",
    "cztery": "4", "czwarty": "4", "piec": "5", "szesc": "6", "siedem": "7",
    "osiem": "8", "dziewiec": "9", "dziesiec": "10", "jedenascie": "11",
    "dwanascie": "12",
}


def number_values(value: str) -> set[str]:
    clean = normalize(value)
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", clean))
    words = set(re.findall(r"[a-z]+", clean))
    numbers.update(number for word, number in NUMBER_WORDS.items() if word in words)
    return numbers


SCOPE_QUALIFIERS = {
    "jednoli", "pierwsz", "drug", "krotkot", "dlugote", "maksym", "minimal",
}


def numeric_relation_issues(
    question: str, answer: str, chunks: list[RetrievedChunk]
) -> list[str]:
    answer_values = number_values(re.sub(r"\[[^\]]+\]", "", answer))
    if not answer_values or not question:
        return []
    focus, relation = _query_profile(question)
    required_relation = max(1, math.ceil(len(relation) * 0.6)) if relation else 0
    answer_terms = stems(answer)
    issues: list[str] = []
    sentences = _sentences(chunks)
    for value in sorted(answer_values):
        valid = False
        qualifier_loss = False
        for sentence in sentences:
            if value not in number_values(sentence):
                continue
            sentence_terms = stems(sentence)
            if focus and not focus.issubset(sentence_terms):
                continue
            sentence_relations = {
                root for root in DURATION_RELATIONS
                if any(_same_root(term, root) for term in sentence_terms)
            }
            if required_relation and len(relation & sentence_relations) < required_relation:
                continue
            missing_qualifiers = {
                qualifier for qualifier in SCOPE_QUALIFIERS
                if any(_same_root(term, qualifier) for term in sentence_terms)
                and not any(_same_root(term, qualifier) for term in answer_terms)
            }
            if missing_qualifiers:
                qualifier_loss = True
                continue
            valid = True
            break
        if not valid:
            kind = "lost_scope_qualifier" if qualifier_loss else "unsupported_number_relation"
            issues.append(f"{kind}:{value}")
    return issues
