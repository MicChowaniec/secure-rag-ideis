from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PIISpan:
    kind: str
    start: int
    end: int
    score: float


@dataclass
class PrivacyResult:
    masked_text: str
    spans: list[PIISpan] = field(default_factory=list)
    backend: str = "regex"

    @property
    def contains_pii(self) -> bool:
        return bool(self.spans)


@dataclass
class SafetyResult:
    category: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    backend: str = "heuristic"
    raw_labels: dict[str, float] = field(default_factory=dict)

    @property
    def unsafe(self) -> bool:
        return self.category != "clean"


@dataclass
class NLPResult:
    entities: list[dict[str, str]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    query_category: str = "inne"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    score: float

    def citation(self) -> str:
        return f"{self.source}, s. {self.page}"


@dataclass
class BotResponse:
    text: str
    status: str
    privacy: PrivacyResult
    input_safety: SafetyResult
    nlp: NLPResult
    sources: list[RetrievedChunk] = field(default_factory=list)
    output_safety: SafetyResult | None = None
    trace_id: str = ""
    generation_backend: str = "not_called"
    grounding_warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def debug_dict(self) -> dict[str, Any]:
        return asdict(self)
