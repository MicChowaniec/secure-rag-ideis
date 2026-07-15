from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_backend: str = "auto"
    ollama_chat_model: str = "qooba/bielik-1.5b-v3.0-instruct:Q8_0"
    ollama_embed_model: str = "nomic-embed-text"
    privacy_model: str = "openai/privacy-filter"
    bielik_guard_model: str = "speakleash/Bielik-Guard-0.1B-v1.1"
    use_privacy_model: bool = False
    use_bielik_guard: bool = False
    pii_policy: str = "mask"
    safety_threshold: float = 0.65
    rag_top_k: int = 3
    rag_min_score: float = 0.08
    vector_db_path: Path = Path("data/vector_store.sqlite3")
    audit_log_path: Path = Path("data/audit.jsonl")
    max_input_chars: int = 4000
    block_prompt_injection: bool = True

    topic: str = "regulamin studiów, zasady zaliczeń i prawa studenta"

    @classmethod
    def from_env(cls) -> "Settings":
        project_env = Path(__file__).resolve().parents[1] / ".env"
        _load_env_file(project_env)
        ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        inferred_backend = "cpu_avx2-fallback" if ollama_url.endswith(":11435") else "auto"
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            ollama_url=ollama_url,
            ollama_backend=os.getenv("OLLAMA_BACKEND", inferred_backend),
            ollama_chat_model=os.getenv(
                "OLLAMA_CHAT_MODEL", "qooba/bielik-1.5b-v3.0-instruct:Q8_0"
            ),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            privacy_model=os.getenv("PRIVACY_MODEL", "openai/privacy-filter"),
            bielik_guard_model=os.getenv(
                "BIELIK_GUARD_MODEL", "speakleash/Bielik-Guard-0.1B-v1.1"
            ),
            use_privacy_model=_bool("USE_PRIVACY_MODEL", False),
            use_bielik_guard=_bool("USE_BIELIK_GUARD", False),
            pii_policy=os.getenv("PII_POLICY", "mask").lower(),
            safety_threshold=float(os.getenv("SAFETY_THRESHOLD", "0.65")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
            rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.08")),
            vector_db_path=Path(os.getenv("VECTOR_DB_PATH", "data/vector_store.sqlite3")),
            audit_log_path=Path(os.getenv("AUDIT_LOG_PATH", "data/audit.jsonl")),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "4000")),
            block_prompt_injection=_bool("BLOCK_PROMPT_INJECTION", True),
        )
