from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import BotResponse


class AuditLogger:
    def __init__(self, path: Path, console: bool = False) -> None:
        self.path = path
        self.console = console
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _short(value: str, limit: int = 260) -> str:
        compact = " ".join((value or "").split())
        return compact if len(compact) <= limit else compact[: limit - 1] + "…"

    @staticmethod
    def _safety_scores(response_safety) -> str:
        categories = (
            "toxic", "spam", "hate_speech", "self_harm", "violence", "sexual"
        )
        scores = response_safety.scores
        harmful_max = max((float(scores.get(name, 0.0)) for name in categories), default=0.0)
        clean_score = (
            response_safety.score
            if response_safety.category == "clean"
            else max(0.0, 1.0 - harmful_max)
        )
        values = [f"{name}={float(scores.get(name, 0.0)):.3f}" for name in categories]
        values.append(f"clean={clean_score:.3f}")
        return ", ".join(values)

    def _write_console(self, response: BotResponse, masked_input: str) -> None:
        pii_types = sorted({span.kind for span in response.privacy.spans})
        pii_action = (
            "BLOCK" if response.status == "blocked_pii"
            else "MASK" if response.privacy.contains_pii
            else "PASS"
        )
        entities = response.nlp.entities or []
        entity_text = ", ".join(
            f"{item.get('label', 'ENTITY')}={self._short(item.get('text', ''), 80)}"
            for item in entities
        ) or "brak"
        keywords = ", ".join(response.nlp.keywords) or "brak"
        retrieval = "; ".join(
            f"{chunk.chunk_id} | {chunk.source}, s. {chunk.page} | score={chunk.score:.4f}"
            for chunk in response.sources
        ) or "POMINIĘTY / brak trafień"
        generation_used_prompt = (
            response.generation_backend.startswith("ollama:")
            or response.generation_backend == "extractive-fallback"
        )
        prompt_state = (
            f"GOTOWY: temat + zasady bezpieczeństwa + {len(response.sources)} fragment(y)"
            if generation_used_prompt else "POMINIĘTY (odpowiedź regułowa lub wcześniejsza blokada)"
        )
        llm_state = (
            f"WYWOŁANY: {response.generation_backend}"
            if response.generation_backend.startswith("ollama:")
            else f"POMINIĘTY / zastąpiony: {response.generation_backend}"
        )
        output_safety = response.output_safety
        output_state = (
            f"{output_safety.category} | confidence={output_safety.score:.4f} | "
            f"backend={output_safety.backend}"
            if output_safety else "POMINIĘTY (wiadomość zatrzymana wcześniej)"
        )

        lines = [
            "",
            "=" * 78,
            f"PRZEBIEG ROZMOWY | trace_id={response.trace_id} | status={response.status}",
            "=" * 78,
            f"[1/9] Telegram Bot -> Input użytkownika: {self._short(masked_input)}",
            f"[2/9] OpenAI Privacy Filter: {pii_action} | found={response.privacy.contains_pii} "
            f"| types={pii_types or ['brak']} | backend={response.privacy.backend}",
            f"[3/9] Bielik Guard: {response.input_safety.category} "
            f"| confidence={response.input_safety.score:.4f} | backend={response.input_safety.backend}",
            f"      scores: {self._safety_scores(response.input_safety)}",
            f"[4/9] NER + keywords: entities={entity_text} | keywords={keywords} "
            f"| category={response.nlp.query_category}",
            f"[5/9] Retriever RAG: {retrieval}",
            f"[6/9] Prompt builder: {prompt_state}",
            f"[7/9] Lokalny LLM przez Ollama: {llm_state}",
            f"[8/9] Output filter: {output_state} "
            f"| grounding={response.grounding_warnings or ['OK / nie dotyczy']}",
            f"[9/9] Odpowiedź w Telegramie: {self._short(response.text, 360)}",
            f"CZASY [ms]: {json.dumps(response.timings_ms, ensure_ascii=False, sort_keys=True)}",
            "=" * 78,
        ]
        print("\n".join(lines), flush=True)

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
            if self.console:
                self._write_console(response, masked_input)
