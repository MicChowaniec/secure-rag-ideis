from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import BotResponse


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, response: BotResponse, masked_input: str) -> None:
        # Nigdy nie zapisujemy surowej wiadomości ani treści wykrytych spanów PII.
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": response.trace_id,
            "status": response.status,
            "masked_input": masked_input,
            "pii": {
                "found": response.privacy.contains_pii,
                "types": sorted({span.kind for span in response.privacy.spans}),
                "backend": response.privacy.backend,
            },
            "safety": asdict(response.input_safety),
            "nlp": asdict(response.nlp),
            "retrieval": [
                {"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "score": c.score}
                for c in response.sources
            ],
            "output_safety": asdict(response.output_safety) if response.output_safety else None,
            "generation_backend": response.generation_backend,
            "grounding_warnings": response.grounding_warnings,
            "timings_ms": response.timings_ms,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
