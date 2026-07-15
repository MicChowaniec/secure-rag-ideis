from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader

from .config import Settings
from .rag import Embedder, VectorStore, split_text


def ingest_pdf(path: Path, store: VectorStore) -> int:
    reader = PdfReader(path)
    records: list[tuple[str, str, str, int]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
        for index, chunk in enumerate(split_text(text)):
            digest = hashlib.sha256(f"{path.name}:{page_number}:{index}:{chunk}".encode()).hexdigest()[:20]
            records.append((digest, chunk, path.name, page_number))
    store.add_many(records)
    store.commit()
    return len(records)


def ingest_text(path: Path, store: VectorStore) -> int:
    """Index a curated website snapshot or other UTF-8 text knowledge source."""
    records: list[tuple[str, str, str, int]] = []
    text = path.read_text(encoding="utf-8")
    for index, chunk in enumerate(split_text(text)):
        digest = hashlib.sha256(f"{path.name}:1:{index}:{chunk}".encode()).hexdigest()[:20]
        records.append((digest, chunk, path.name, 1))
    store.add_many(records)
    store.commit()
    return len(records)


def ingest_path(path: Path, store: VectorStore) -> int:
    return ingest_pdf(path, store) if path.suffix.casefold() == ".pdf" else ingest_text(path, store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indeksowanie dokumentów PDF i źródeł tekstowych")
    defaults = [
        *Path("data/pdf").glob("*.pdf"),
        *Path("data/knowledge").glob("*.md"),
        *Path("data/knowledge").glob("*.txt"),
    ]
    parser.add_argument("paths", nargs="*", type=Path, default=defaults)
    parser.add_argument("--append", action="store_true", help="Nie czyść istniejącego indeksu")
    args = parser.parse_args()
    settings = Settings.from_env()
    store = VectorStore(
        settings.vector_db_path,
        Embedder(settings.ollama_url, settings.ollama_embed_model),
    )
    try:
        if not args.append:
            store.clear()
        total = sum(ingest_path(path, store) for path in args.paths)
        print(f"Zindeksowano {total} fragmentów z {len(args.paths)} źródeł.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
