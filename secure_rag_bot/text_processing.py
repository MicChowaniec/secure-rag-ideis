from __future__ import annotations

import re
import unicodedata


EMBEDDING_NORMALIZATION_VERSION = "normalized-v1"


def normalize_unicode(value: str) -> str:
    """Normalize Unicode and remove control characters without losing line structure."""
    normalized = unicodedata.normalize("NFKC", value or "").replace("\u00ad", "")
    return "".join(
        character
        for character in normalized
        if character in "\n\t" or ord(character) >= 32
    )


def normalize_user_text(value: str) -> str:
    return normalize_unicode(value).strip()


def clean_pdf_text(value: str) -> str:
    """Repair common PDF extraction artifacts while preserving the source wording."""
    text = normalize_unicode(value)
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\r?\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_embedding(value: str) -> str:
    """Return one shared representation for document and query embeddings."""
    text = clean_pdf_text(value).casefold()
    text = re.sub(r"\b(?:studia|studiow|studiów)\s+i\s+stopnia\b", "studia pierwszego stopnia", text)
    text = re.sub(r"\b(?:studia|studiow|studiów)\s+ii\s+stopnia\b", "studia drugiego stopnia", text)
    return text


def normalize_for_lexical(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", normalize_unicode(value).casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    # Unicode NFKD nie rozkłada polskiego „ł”, dlatego mapujemy je jawnie.
    return without_marks.translate(str.maketrans({"ł": "l"}))


def lexical_stems(value: str, minimum_length: int = 3, prefix_length: int = 5) -> set[str]:
    words = re.findall(rf"[a-z]{{{minimum_length},}}", normalize_for_lexical(value))
    return {word[:prefix_length] for word in words}


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = sum(a != b for a, b in zip(left, right))
        return differences <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index_short = index_long = differences = 0
    while index_short < len(shorter) and index_long < len(longer):
        if shorter[index_short] == longer[index_long]:
            index_short += 1
            index_long += 1
        else:
            differences += 1
            index_long += 1
            if differences > 1:
                return False
    return True


def has_approximate_root(value: str, target: str, prefix_length: int = 5) -> bool:
    """Match a domain term with tolerance for one typo in its stable prefix."""
    target_root = normalize_for_lexical(target)[:prefix_length]
    return any(
        _edit_distance_at_most_one(word, target_root)
        for word in lexical_stems(value, prefix_length=prefix_length)
    )


def split_sentences(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?;])\s+", normalize_unicode(value))
        if part.strip()
    ]
