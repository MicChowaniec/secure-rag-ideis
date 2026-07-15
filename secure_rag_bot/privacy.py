from __future__ import annotations

import re
from collections.abc import Iterable

from .models import PIISpan, PrivacyResult


PATTERNS: dict[str, re.Pattern[str]] = {
    "private_email": re.compile(
        r"(?<![\w.-])[\w.+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w-])"
    ),
    "private_phone": re.compile(r"(?<!\d)(?:\+?48[ -]?)?(?:\d[ -]?){9}(?!\d)"),
    "account_number": re.compile(r"(?<!\d)(?:\d[ -]?){16,26}(?!\d)"),
    "private_person": re.compile(
        r"\b(?:nazywam się|jestem|imię(?: i nazwisko)?(?: to|:))\s+"
        r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)",
        re.IGNORECASE,
    ),
    "secret": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|(?:hasło|password|token|api[_ -]?key)\s*[:=]\s*\S+)",
        re.IGNORECASE,
    ),
}

MASKS = {
    "private_email": "[EMAIL]",
    "private_phone": "[TELEFON]",
    "account_number": "[NUMER_KONTA_LUB_KARTY]",
    "private_person": "[OSOBA_PRYWATNA]",
    "private_address": "[ADRES]",
    "private_url": "[URL_PRYWATNY]",
    "private_date": "[DATA_PRYWATNA]",
    "secret": "[SEKRET]",
}


class PrivacyFilter:
    """Lokalny filtr PII. Model HF jest opcjonalny, regex stanowi warstwę zapasową."""

    def __init__(self, model_name: str, use_model: bool = False) -> None:
        self.model_name = model_name
        self.use_model = use_model
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
            )
        return self._pipeline

    @staticmethod
    def _regex_spans(text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                start, end = match.span(1) if match.lastindex else match.span()
                candidate = re.sub(r"\D", "", text[start:end])
                if kind == "account_number" and len(candidate) < 16:
                    continue
                spans.append(PIISpan(kind, start, end, 1.0))
        return spans

    @staticmethod
    def _dedupe(spans: Iterable[PIISpan]) -> list[PIISpan]:
        selected: list[PIISpan] = []
        for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score)):
            if not any(span.start < old.end and old.start < span.end for old in selected):
                selected.append(span)
        return sorted(selected, key=lambda s: s.start)

    @staticmethod
    def _mask(text: str, spans: list[PIISpan]) -> str:
        out = text
        for span in reversed(spans):
            out = out[: span.start] + MASKS.get(span.kind, "[DANE_WRAŻLIWE]") + out[span.end :]
        return out

    def analyze(self, text: str, use_model: bool | None = None) -> PrivacyResult:
        spans = self._regex_spans(text)
        backend = "regex"
        model_enabled = self.use_model if use_model is None else use_model
        if model_enabled:
            try:
                predictions = self._load()(text)
                spans.extend(
                    PIISpan(
                        str(p.get("entity_group", p.get("entity", "private_person"))).lower(),
                        int(p["start"]),
                        int(p["end"]),
                        float(p["score"]),
                    )
                    for p in predictions
                    if float(p.get("score", 0)) >= 0.50
                )
                backend = self.model_name
            except (ImportError, OSError, RuntimeError, ValueError):
                backend = "regex-fallback"
        final = self._dedupe(spans)
        return PrivacyResult(self._mask(text, final), final, backend)
