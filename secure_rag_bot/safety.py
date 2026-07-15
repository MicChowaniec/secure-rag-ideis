from __future__ import annotations

import re

from .models import SafetyResult
from .text_processing import normalize_for_lexical


NORMALIZE = {
    "hate": "hate_speech",
    "hate/aggression": "hate_speech",
    "vulgar": "toxic",
    "vulgarities": "toxic",
    "sex": "sexual",
    "sexual content": "sexual",
    "self-harm": "self_harm",
    "self_harm": "self_harm",
    "crime": "violence",
}

HEURISTICS: dict[str, tuple[str, ...]] = {
    "self_harm": ("zabić się", "samobój", "samookalecz", "nie chcę żyć"),
    "violence": ("zabić", "bomba", "pobić", "broń", "wysadzić"),
    "hate_speech": ("nienawidzę wszystkich", "podludzie", "wytępić"),
    "sexual": ("pornograf", "seksualn", "nagość", "erotycz"),
    "toxic": ("idiota", "debil", "kretyn"),
}


class SafetyClassifier:
    """Bielik Guard + jawne uzupełnienie etykiety spam, której model nie obsługuje."""

    def __init__(self, model_name: str, threshold: float = 0.65, use_model: bool = False) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.use_model = use_model
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("text-classification", model=self.model_name, top_k=None)
        return self._pipeline

    @staticmethod
    def _spam_score(text: str) -> float:
        links = len(re.findall(r"https?://|www\.", text, re.I))
        repeated = bool(re.search(r"(.{3,})\1{2,}", text, re.I))
        shouting = len(text) > 15 and sum(c.isupper() for c in text) / max(len(text), 1) > 0.65
        return min(0.99, links * 0.25 + repeated * 0.45 + shouting * 0.35)

    @staticmethod
    def _contains_heuristic(text: str, term: str) -> bool:
        normalized_term = normalize_for_lexical(term)
        if " " in normalized_term:
            return normalized_term in text
        words = re.findall(r"[a-z]+", text)
        if len(normalized_term) <= 5:
            return normalized_term in words
        return any(word.startswith(normalized_term) for word in words)

    def analyze(self, text: str) -> SafetyResult:
        scores = {name: 0.0 for name in (*HEURISTICS, "spam")}
        lower = normalize_for_lexical(text)
        for category, terms in HEURISTICS.items():
            hits = sum(self._contains_heuristic(lower, term) for term in terms)
            scores[category] = min(0.99, hits * 0.75)
        scores["spam"] = self._spam_score(text)
        raw: dict[str, float] = {}
        backend = "heuristic"

        if self.use_model:
            try:
                output = self._load()(text)
                rows = output[0] if output and isinstance(output[0], list) else output
                for row in rows:
                    label = str(row["label"]).strip("[]").lower()
                    raw[label] = float(row["score"])
                    normalized = NORMALIZE.get(label)
                    if normalized:
                        scores[normalized] = max(scores.get(normalized, 0), float(row["score"]))
                backend = self.model_name + "+spam-rule"
            except (ImportError, OSError, RuntimeError, ValueError):
                backend = "heuristic-fallback"

        category, score = max(scores.items(), key=lambda item: item[1])
        if score < self.threshold:
            category = "clean"
            score = max(0.0, 1.0 - score)
        return SafetyResult(category, round(score, 4), scores, backend, raw)
