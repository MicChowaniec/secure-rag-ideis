from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from .config import Settings
from .factory import create_pipeline


def _ollama_models(url: str) -> set[str]:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
        payload = json.load(response)
    return {item["name"] for item in payload.get("models", [])}


def _ollama_generation(url: str, model: str) -> str:
    body = json.dumps({
        "model": model, "prompt": "Odpowiedz: OK", "stream": False,
        "options": {"temperature": 0, "num_predict": 3},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return str(json.load(response).get("response", "")).strip()


def main() -> None:
    settings = Settings.from_env()
    checks: dict[str, object] = {}
    errors: list[str] = []
    checks["ollama_backend"] = settings.ollama_backend
    checks["ollama_url"] = settings.ollama_url

    try:
        models = _ollama_models(settings.ollama_url)
        checks["ollama"] = "ok"
        checks["ollama_models"] = sorted(models)
        for required in (settings.ollama_chat_model, settings.ollama_embed_model):
            if required not in models and f"{required}:latest" not in models:
                errors.append(f"Brak modelu Ollama: {required}")
        probe = _ollama_generation(settings.ollama_url, settings.ollama_chat_model)
        checks["ollama_generation"] = probe or "empty"
        if not probe:
            errors.append("Ollama zwróciła pustą próbę generacji")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        checks["ollama"] = f"error: {type(exc).__name__}"
        errors.append("Serwer Ollama nie odpowiada pod " + settings.ollama_url)

    pipeline = create_pipeline(settings)
    rag_chunks = pipeline.store.count()
    checks["rag_chunks"] = rag_chunks
    if not rag_chunks:
        errors.append("Baza RAG jest pusta")

    privacy = pipeline.privacy.analyze("Nazywam się Jan Kowalski, e-mail jan@example.com")
    checks["privacy_backend"] = privacy.backend
    checks["privacy_types"] = sorted({span.kind for span in privacy.spans})
    if settings.use_privacy_model and privacy.backend != settings.privacy_model:
        errors.append("Privacy Filter nie został załadowany; działa fallback")

    safety = pipeline.safety.analyze("To jest zwykłe pytanie o egzamin.")
    checks["guard_backend"] = safety.backend
    if settings.use_bielik_guard and not safety.backend.startswith(settings.bielik_guard_model):
        errors.append("Bielik Guard nie został załadowany; działa fallback")

    pipeline.store.close()

    result = {"ok": not errors, "checks": checks, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
