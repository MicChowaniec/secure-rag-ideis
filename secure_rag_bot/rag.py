from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

from .models import RetrievedChunk
from .text_processing import (
    EMBEDDING_NORMALIZATION_VERSION,
    clean_pdf_text,
    normalize_for_embedding,
    normalize_for_lexical,
    has_approximate_root,
)


class Embedder:
    def __init__(self, ollama_url: str, model: str, dimensions: int = 384) -> None:
        self.ollama_url = ollama_url
        self.model = model
        self.dimensions = dimensions

    def hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = normalize_for_embedding(text)
        tokens = re.findall(r"[a-z0-9]+", normalize_for_lexical(normalized))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, text: str) -> tuple[list[float], str]:
        vectors, backend = self.embed_many([text])
        return vectors[0], backend

    def embed_many(self, texts: list[str]) -> tuple[list[list[float]], str]:
        normalized_texts = [normalize_for_embedding(text) for text in texts]
        payload = json.dumps({"model": self.model, "input": normalized_texts}).encode()
        request = urllib.request.Request(
            f"{self.ollama_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.load(response)
            embeddings = data["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError("Ollama zwróciła inną liczbę embeddingów niż wejść")
            return embeddings, f"ollama:{self.model}:{EMBEDDING_NORMALIZATION_VERSION}"
        except (
            urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError,
            ValueError, json.JSONDecodeError,
        ):
            return (
                [self.hash_embed(text) for text in normalized_texts],
                f"hash-fallback:{EMBEDDING_NORMALIZATION_VERSION}",
            )


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.path = path
        self.embedder = embedder
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,
            page INTEGER NOT NULL, vector TEXT NOT NULL, embedding_backend TEXT NOT NULL)"""
        )
        self.db.commit()

    def count(self) -> int:
        with self._lock:
            return int(self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def clear(self) -> None:
        with self._lock:
            self.db.execute("DELETE FROM chunks")
            self.db.commit()

    def add(self, chunk_id: str, text: str, source: str, page: int) -> None:
        self.add_many([(chunk_id, text, source, page)])

    def add_many(self, records: list[tuple[str, str, str, int]], batch_size: int = 16) -> None:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors, backend = self.embedder.embed_many([record[1] for record in batch])
            rows = [
                (chunk_id, text, source, page, json.dumps(vector), backend)
                for (chunk_id, text, source, page), vector in zip(batch, vectors)
            ]
            with self._lock:
                self.db.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)

    def commit(self) -> None:
        with self._lock:
            self.db.commit()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0

    @staticmethod
    def _terms(value: str) -> list[str]:
        stop = {
            "który", "która", "według", "regulaminu", "studiów", "mojego", "nazywam",
            "student", "studenta", "może", "mogę", "jakie", "jaka", "oraz", "jest",
            "email", "osoba", "prywatna", "proszę", "pytanie",
        }
        words = re.findall(r"[a-z]{3,}", normalize_for_lexical(value))
        return [word[:5] for word in words if word not in stop]

    @staticmethod
    def _expand_query(query: str) -> str:
        lower = normalize_for_lexical(query)
        additions: list[str] = []
        if "egzamin" in lower and "popraw" in lower:
            additions.append("jedna próba poprawy zaliczenia termin egzaminacyjny ocena niedostateczna")
        if "odwol" in lower:
            additions.append(
                "decyzja administracyjna kodeks postępowania administracyjnego "
                "ponowne rozpatrzenie sprawy rektor"
            )
        if "urlop" in lower:
            additions.append("urlop zdrowotny losowy naukowy pisemny wniosek")
        if "praw" in lower and "student" in lower:
            additions.append(
                "student ma prawo przenoszenia uznawania punktów ECTS indywidualnej "
                "organizacji urlopów egzaminu komisyjnego powtarzania zajęć"
            )
        if has_approximate_root(query, "dyplom"):
            additions.append(
                "absolwent otrzymuje dyplom ukończenia studiów uczelnia wydaje dyplom "
                "w terminie 30 dni suplement odpisy"
            )
        if any(term in lower for term in ("platn", "odplat", "czesn", "oplat")):
            additions.append(
                "nauka w uczelni jest odpłatna warunki odpłatności umowa o świadczenie "
                "usług edukacyjnych regulamin opłat"
            )
        if "zapis" in lower and "przedm" in lower:
            additions.append(
                "szczegółowe zasady zapisów zakresy kształcenia grupy zajęć do wyboru "
                "zarządzenie Rektora"
            )
        if any(term in lower for term in ("krakow", "filia", "dziekanat", "kampus")):
            additions.append(
                "Uniwersytet DSW Ideis Kraków ul. Św. Filipa 17 dziekanat oficjalna strona "
                "krakowska lokalizacja"
            )
        return query + " " + " ".join(additions)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        with self._lock:
            rows = list(self.db.execute(
                "SELECT id, text, source, page, vector, embedding_backend FROM chunks"
            ))
        expanded_query = self._expand_query(query)
        query_terms = set(self._terms(expanded_query))
        document_terms = {row[0]: self._terms(row[1]) for row in rows}
        document_count = max(len(rows), 1)
        average_length = sum(len(terms) for terms in document_terms.values()) / document_count or 1.0
        document_frequency = {
            term: sum(term in set(terms) for terms in document_terms.values()) for term in query_terms
        }
        bm25: dict[str, float] = {}
        for chunk_id, terms in document_terms.items():
            term_counts = {term: terms.count(term) for term in query_terms}
            length_norm = 1.5 * (1 - 0.75 + 0.75 * len(terms) / average_length)
            score = 0.0
            for term, frequency in term_counts.items():
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                score += inverse_frequency * (frequency * 2.5) / (frequency + length_norm)
            bm25[chunk_id] = score
        max_bm25 = max(bm25.values(), default=1.0) or 1.0

        query_vectors: dict[str, list[float]] = {}
        for chunk_id, text, source, page, raw_vector, backend in rows:
            if backend not in query_vectors:
                query_vectors[backend] = (
                    self.embedder.hash_embed(expanded_query)
                    if backend.startswith("hash-fallback")
                    else self.embedder.embed(expanded_query)[0]
                )
            semantic = max(0.0, self._cosine(query_vectors[backend], json.loads(raw_vector)))
            lexical = bm25.get(chunk_id, 0.0) / max_bm25
            score = 0.30 * semantic + 0.70 * lexical
            chunks.append(RetrievedChunk(chunk_id, text, source, page, score))
        ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
        selected: list[RetrievedChunk] = []
        seen_pages: set[tuple[str, int]] = set()
        for chunk in ranked:
            page_key = (chunk.source, chunk.page)
            if page_key in seen_pages:
                continue
            selected.append(chunk)
            seen_pages.add(page_key)
            if len(selected) == top_k:
                break
        if len(selected) < top_k:
            selected_ids = {chunk.chunk_id for chunk in selected}
            selected.extend(chunk for chunk in ranked if chunk.chunk_id not in selected_ids)
        return selected[:top_k]


def split_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    clean = clean_pdf_text(text)
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            boundary = clean.rfind(". ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks
