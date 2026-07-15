import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from secure_rag_bot.audit import AuditLogger
from secure_rag_bot.models import (
    BotResponse,
    NLPResult,
    PIISpan,
    PrivacyResult,
    RetrievedChunk,
    SafetyResult,
)


class TerminalTraceTests(unittest.TestCase):
    def test_trace_shows_pipeline_without_raw_pii(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = AuditLogger(Path(temp) / "audit.jsonl", console=True)
            response = BotResponse(
                text="Student ma prawo do egzaminu poprawkowego.",
                status="answered",
                privacy=PrivacyResult(
                    "Mój e-mail [EMAIL]. Czy mam egzamin poprawkowy?",
                    [PIISpan("private_email", 11, 26, 1.0)],
                    "openai/privacy-filter",
                ),
                input_safety=SafetyResult(
                    "clean",
                    0.99,
                    {"toxic": 0.01, "spam": 0.0},
                    "speakleash/Bielik-Guard-0.1B-v1.1+spam-rule",
                ),
                nlp=NLPResult([], ["egzamin", "poprawkowy"], "zaliczenia_i_egzaminy"),
                sources=[RetrievedChunk("c1", "tekst", "regulamin.pdf", 3, 0.91)],
                output_safety=SafetyResult("clean", 1.0, {}, "heuristic"),
                trace_id="trace-test",
                generation_backend="ollama:bielik-test",
                timings_ms={"total": 12.5},
            )

            terminal = io.StringIO()
            with redirect_stdout(terminal):
                logger.write(response, response.privacy.masked_text)

            output = terminal.getvalue()
            self.assertIn("[1/9] Telegram Bot", output)
            self.assertIn("[2/9] OpenAI Privacy Filter: MASK", output)
            self.assertIn("[3/9] Bielik Guard: clean", output)
            self.assertIn("[5/9] Retriever RAG", output)
            self.assertIn("[7/9] Lokalny LLM przez Ollama: WYWOŁANY", output)
            self.assertIn("[9/9] Odpowiedź w Telegramie", output)
            self.assertIn("[EMAIL]", output)
            self.assertNotIn("ala@example.com", output)


if __name__ == "__main__":
    unittest.main()
