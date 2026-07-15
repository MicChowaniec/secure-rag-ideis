from __future__ import annotations

import uuid
import re
from time import perf_counter

from .audit import AuditLogger
from .config import Settings
from .llm import OllamaLLM
from .grounding import grounded_fallback, grounding_issues
from .fact_policy import answer_from_fact_policy
from .models import BotResponse, NLPResult, PrivacyResult, SafetyResult
from .nlp import NLPAnalyzer
from .privacy import PrivacyFilter
from .rag import VectorStore
from .safety import SafetyClassifier
from .text_processing import normalize_for_lexical, normalize_user_text
from .evidence import (
    duration_answer_from_evidence,
    is_study_domain_query,
    quantitative_evidence_available,
)


ABUSE_PATTERNS = (
    r"ignoruj (?:wszystkie |poprzednie )?(?:instrukcje|polecenia)",
    r"ujawnij (?:prompt|instrukcję systemową|system prompt)",
    r"(?:system prompt|prompt systemowy)",
    r"(?:jailbreak|tryb dan\b|developer mode)",
    r"udawaj,? że (?:nie masz|jesteś)",
    r"wykonaj polecenie (?:z|w) (?:pdf|kontekście|dokumencie)",
)

CONTACT_LOCATION_PATTERNS = (
    r"\b(?:jaki|podaj|znasz)\b.{0,30}\badres(?:em|u)?\b.{0,20}\buczelni\b",
    r"\b(?:gdzie|dokąd)\b.{0,35}\b(?:sekretariat|dziekanat|uczelnia|kampus)\b",
    r"\b(?:telefon|numer telefonu|e-?mail)\b.{0,25}\b(?:sekretariatu|dziekanatu|uczelni)\b",
)

UNSAFE_MESSAGES = {
    "self_harm": "Jeśli grozi Ci bezpośrednie niebezpieczeństwo, zadzwoń pod 112. Nie mogę pomagać w wyrządzaniu krzywdy. Mogę natomiast pomóc znaleźć bezpieczne wsparcie na uczelni.",
    "violence": "Nie mogę pomóc w działaniach przestępczych ani przemocy.",
    "hate_speech": "Nie mogę wspierać mowy nienawiści ani agresji wobec osób lub grup.",
    "sexual": "Nie mogę pomóc w tej treści. Mogę odpowiadać na pytania dotyczące regulaminu studiów.",
    "toxic": "Spróbujmy kontynuować bez obrażania innych. Mogę pomóc w sprawach regulaminu studiów.",
    "spam": "Wiadomość wygląda na spam, dlatego nie została przekazana dalej.",
}


class SecureRAGPipeline:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        privacy: PrivacyFilter | None = None,
        safety: SafetyClassifier | None = None,
        nlp: NLPAnalyzer | None = None,
        llm: OllamaLLM | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.privacy = privacy or PrivacyFilter(settings.privacy_model, settings.use_privacy_model)
        self.safety = safety or SafetyClassifier(
            settings.bielik_guard_model, settings.safety_threshold, settings.use_bielik_guard
        )
        self.nlp = nlp or NLPAnalyzer()
        self.llm = llm or OllamaLLM(settings.ollama_url, settings.ollama_chat_model)
        self.audit = audit or AuditLogger(
            settings.audit_log_path, console=settings.terminal_trace
        )

    def _finish(self, response: BotResponse) -> BotResponse:
        self.audit.write(response, response.privacy.masked_text)
        return response

    @staticmethod
    def _normalize_input(text: str) -> str:
        return normalize_user_text(text)

    @staticmethod
    def _is_prompt_injection(text: str) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in ABUSE_PATTERNS)

    @staticmethod
    def _is_contact_location_query(text: str) -> bool:
        normalized = normalize_for_lexical(text)
        institution = any(
            term in normalized for term in ("uczeln", "dziekanat", "sekretariat", "kampus")
        )
        return institution and (
            "adres" in normalized
            or "gdzie" in normalized
            or "dokad" in normalized
            or "telefon" in normalized
            or "numer telefonu" in normalized
            or "email" in normalized
        )

    @staticmethod
    def _is_capability_query(text: str) -> bool:
        normalized = normalize_for_lexical(text)
        return any(
            phrase in normalized
            for phrase in (
                "na jakie pytania", "co potrafisz", "w czym pomozesz",
                "jaki masz zakres", "o czym mozna", "czym sie zajmujesz",
            )
        )

    @staticmethod
    def is_followup_query(text: str, conversation_context: str = "") -> bool:
        """Recognize a short anaphoric question without guessing a new conversation topic."""
        if not conversation_context.strip():
            return False
        normalized = normalize_for_lexical(text)
        words = normalized.split()
        explicit_subjects = (
            "studi", "dypl", "urlop", "egzamin", "przedm", "oplat", "czesn",
            "skresl", "dziekan", "zalicz", "prakty", "promotor", "semestr", "ects",
        )
        if normalized.startswith(("a ", "ale ", "okej a ", "ok a ")):
            return True
        if "odwol" in normalized and not any(
            root in normalized for root in ("skresl", "oplat", "decyz", "dziekan")
        ):
            return True
        has_explicit_subject = any(root in normalized for root in explicit_subjects)
        interrogative = any(
            normalized.startswith(prefix)
            for prefix in ("jak ", "ile ", "kiedy ", "gdzie ", "czy ", "co ", "na czym ")
        )
        return len(words) <= 7 and interrogative and not has_explicit_subject

    def handle(self, text: str, conversation_context: str = "") -> BotResponse:
        request_started = perf_counter()
        timings: dict[str, float] = {}

        def finish(response: BotResponse) -> BotResponse:
            response.timings_ms = {
                **timings,
                "total": round((perf_counter() - request_started) * 1000, 2),
            }
            return self._finish(response)

        trace_id = uuid.uuid4().hex[:12]
        stage_started = perf_counter()
        text = self._normalize_input(text or "")
        timings["normalize"] = round((perf_counter() - stage_started) * 1000, 2)
        clean = SafetyResult("clean", 1.0)
        empty_privacy = PrivacyResult("")
        empty_nlp = NLPResult()
        if not text:
            return finish(BotResponse(
                "Wiadomość jest pusta. Zadaj pytanie dotyczące regulaminu studiów.",
                "invalid_input", empty_privacy, clean, empty_nlp, trace_id=trace_id,
            ))
        if len(text) > self.settings.max_input_chars:
            privacy = PrivacyResult("[WIADOMOŚĆ_ZBYT_DŁUGA]", backend="length-guard")
            return finish(BotResponse(
                f"Wiadomość jest za długa. Limit wynosi {self.settings.max_input_chars} znaków.",
                "invalid_input", privacy, clean, empty_nlp, trace_id=trace_id,
            ))
        raw_prompt_injection = self._is_prompt_injection(text)
        stage_started = perf_counter()
        privacy = self.privacy.analyze(text)
        timings["privacy_input"] = round((perf_counter() - stage_started) * 1000, 2)
        safe_text = privacy.masked_text
        stage_started = perf_counter()
        nlp = self.nlp.analyze(safe_text)
        timings["ner_keywords"] = round((perf_counter() - stage_started) * 1000, 2)

        if privacy.contains_pii and self.settings.pii_policy == "block":
            return finish(BotResponse(
                "Wiadomość zawiera dane wrażliwe i została zablokowana. Usuń je i spróbuj ponownie.",
                "blocked_pii", privacy, clean, nlp, trace_id=trace_id,
            ))

        stage_started = perf_counter()
        input_safety = self.safety.analyze(safe_text)
        timings["safety_input"] = round((perf_counter() - stage_started) * 1000, 2)
        if input_safety.unsafe:
            return finish(BotResponse(
                UNSAFE_MESSAGES.get(input_safety.category, "Wiadomość została zablokowana."),
                "blocked_unsafe", privacy, input_safety, nlp, trace_id=trace_id,
            ))

        if self.settings.block_prompt_injection and (
            raw_prompt_injection or self._is_prompt_injection(safe_text)
        ):
            return finish(BotResponse(
                "Wiadomość wygląda na próbę zmiany zasad działania asystenta. Mogę odpowiadać wyłącznie na zwykłe pytania dotyczące regulaminu studiów.",
                "blocked_abuse", privacy, input_safety, nlp, trace_id=trace_id,
            ))

        if self._is_capability_query(safe_text):
            return finish(BotResponse(
                "Odpowiadam na pytania o regulamin studiów IDEIS, m.in. o zaliczenia i "
                "egzaminy, urlopy, prawa i obowiązki studenta, skreślenie z listy, dyplom, "
                "opłaty oraz organizację studiów. Jeśli regulamin nie zawiera odpowiedzi "
                "(np. adresu dziekanatu), powiem to wprost zamiast zgadywać.",
                "answered_meta", privacy, input_safety, nlp, trace_id=trace_id,
            ))

        safe_context = self._normalize_input(conversation_context or "")[:1200]
        use_context = self.is_followup_query(safe_text, safe_context)
        query_text = (
            f"Kontekst poprzednich pytań: {safe_context}\nBieżące pytanie: {safe_text}"
            if use_context
            else safe_text
        )

        domain_match = is_study_domain_query(query_text) or self._is_contact_location_query(safe_text)
        if not domain_match:
            return finish(BotResponse(
                "Odpowiadam wyłącznie na pytania o regulamin studiów, zaliczenia i prawa studenta. W dostarczonych materiałach nie znalazłem podstawy do odpowiedzi.",
                "out_of_scope", privacy, input_safety, nlp, trace_id=trace_id,
            ))

        stage_started = perf_counter()
        # Szersza pula służy regułom faktograficznym; do promptu LLM trafia nadal tylko
        # skonfigurowane top-k, więc nie zwiększamy bez potrzeby kontekstu generacyjnego.
        candidates = self.store.search(query_text, max(self.settings.rag_top_k, 8))
        chunks = candidates[: self.settings.rag_top_k]
        timings["retrieval"] = round((perf_counter() - stage_started) * 1000, 2)
        best_score = chunks[0].score if chunks else 0.0
        if not chunks or best_score < self.settings.rag_min_score:
            return finish(BotResponse(
                "Pytanie dotyczy studiów, ale w dostarczonym regulaminie nie znalazłem wystarczającej podstawy do odpowiedzi.",
                "insufficient_context", privacy, input_safety, nlp, chunks, trace_id=trace_id,
            ))

        fact_answer = answer_from_fact_policy(query_text, candidates)
        if self._is_contact_location_query(safe_text) and not fact_answer:
            return finish(BotResponse(
                "W zindeksowanych źródłach nie znalazłem zweryfikowanych danych kontaktowych "
                "dla wskazanej jednostki. Nie będę zgadywać adresu ani telefonu.",
                "insufficient_context", privacy, input_safety, nlp, chunks, trace_id=trace_id,
            ))
        if not fact_answer and not quantitative_evidence_available(query_text, candidates):
            normalized_query = normalize_for_lexical(query_text)
            requested_type = any(
                root in normalized_query for root in ("inzyn", "magister", "licenc")
            )
            detail = (
                " Długość zależy od konkretnego kierunku i programu studiów; podaj kierunek, "
                "a można ją sprawdzić w jego programie lub na oficjalnej stronie."
                if requested_type
                else ""
            )
            return finish(BotResponse(
                "Pytanie dotyczy studiów, ale regulamin nie wiąże wskazanego rodzaju studiów "
                "z konkretną liczbą semestrów. Nie będę łączyć niezależnych fragmentów ani "
                f"zgadywać na podstawie wiedzy ogólnej.{detail}",
                "insufficient_context", privacy, input_safety, nlp, chunks, trace_id=trace_id,
            ))

        stage_started = perf_counter()
        duration_answer = None if fact_answer else duration_answer_from_evidence(query_text, candidates)
        response_status = "answered"
        if fact_answer:
            answer = fact_answer.text
            chunks = fact_answer.sources
            _backend = "deterministic-fact-policy"
            response_status = fact_answer.status
            timings["generation"] = round((perf_counter() - stage_started) * 1000, 2)
        elif duration_answer:
            answer, chunks = duration_answer
            _backend = "extractive-duration-policy"
            timings["generation"] = 0.0
        else:
            answer, _backend = self.llm.generate(query_text, chunks)
            timings["generation"] = round((perf_counter() - stage_started) * 1000, 2)
        answer = (answer or "").strip()
        if not answer:
            return finish(BotResponse(
                "Model lokalny nie zwrócił odpowiedzi. Spróbuj ponownie za chwilę.",
                "model_error", privacy, input_safety, nlp, chunks, trace_id=trace_id,
            ))
        grounding_warnings = (
            [] if _backend in {
                "extractive-fallback", "extractive-duration-policy", "evaluation-stub",
                "deterministic-fact-policy",
            }
            else grounding_issues(answer, chunks, query_text)
        )
        grounding_fallback_used = bool(grounding_warnings)
        if grounding_fallback_used:
            answer = grounded_fallback(chunks)
        stage_started = perf_counter()
        # Pełny model prywatności działa na wejściu. Na wyjściu wystarcza szybka,
        # deterministyczna kontrola PII, po której nadal działa pełny Bielik Guard.
        output_privacy = self.privacy.analyze(answer, use_model=False)
        timings["privacy_output_regex"] = round((perf_counter() - stage_started) * 1000, 2)
        answer = output_privacy.masked_text
        stage_started = perf_counter()
        output_safety = self.safety.analyze(answer)
        timings["safety_output"] = round((perf_counter() - stage_started) * 1000, 2)
        if output_safety.unsafe:
            answer = "Odpowiedź została zatrzymana przez filtr bezpieczeństwa. Spróbuj przeformułować pytanie."
            status = "blocked_output"
        else:
            status = "grounding_fallback" if grounding_fallback_used else response_status
            citations = " ".join(f"[{chunk.citation()}]" for chunk in chunks[:2])
            if (
                chunks
                and not grounding_fallback_used
                and not any(chunk.source in answer for chunk in chunks)
            ):
                answer = f"{answer}\n\nŹródła: {citations}"
            answer = answer[:3900]
        return finish(BotResponse(
            answer, status, privacy, input_safety, nlp, chunks, output_safety, trace_id,
            generation_backend=_backend,
            grounding_warnings=grounding_warnings,
        ))
