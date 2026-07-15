from __future__ import annotations

import re
from collections import Counter

from .models import NLPResult
from .text_processing import normalize_for_lexical


STOP = {
    "oraz", "który", "która", "jest", "jakie", "jaki", "mogę", "może", "tego", "przez",
    "student", "studenta", "studiów", "proszę", "czy", "się", "dla", "nie", "mam", "ile",
}

CATEGORIES = {
    "zaliczenia_i_egzaminy": ("egzamin", "zaliczen", "ocen", "poprawk"),
    "prawa_studenta": ("prawo", "odwoł", "skarg", "samorząd"),
    "urlopy_i_skreslenia": ("urlop", "skreśl", "wznow"),
    "organizacja_studiow": ("ects", "semestr", "dziekan", "program", "zaję"),
    "praca_dyplomowa": ("dyplom", "promotor", "obron"),
}


class NLPAnalyzer:
    def analyze(self, text: str) -> NLPResult:
        entities: list[dict[str, str]] = []
        if "[OSOBA_PRYWATNA]" in text:
            entities.append({"text": "[OSOBA_PRYWATNA]", "label": "PERSON_MASKED"})
        for value in re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text):
            entities.append({"text": value, "label": "DATE"})
        for value in re.findall(r"\b(?:Akademia|Uniwersytet|Politechnika)\s+[\wĄĆĘŁŃÓŚŹŻąćęłńóśźż -]+", text):
            entities.append({"text": value.strip(), "label": "ORG"})
        for value in re.findall(
            r"\b(?:Warszawa|Kraków|Gdańsk|Wrocław|Poznań|Katowice|Dąbrowa Górnicza)\b",
            text,
            re.IGNORECASE,
        ):
            entities.append({"text": value, "label": "LOCATION"})

        words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", text.lower())
        keywords = [word for word, _ in Counter(w for w in words if w not in STOP).most_common(6)]
        lower = normalize_for_lexical(text)
        category = "inne"
        best = 0
        for name, stems in CATEGORIES.items():
            score = sum(normalize_for_lexical(stem) in lower for stem in stems)
            if score > best:
                category, best = name, score
        return NLPResult(entities, keywords, category)
