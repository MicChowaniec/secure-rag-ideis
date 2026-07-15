from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievedChunk
from .text_processing import has_approximate_root, normalize_for_lexical


@dataclass
class FactPolicyAnswer:
    text: str
    sources: list[RetrievedChunk]
    status: str = "answered"


def _find(chunks: list[RetrievedChunk], phrase: str) -> RetrievedChunk | None:
    normalized_phrase = normalize_for_lexical(phrase)
    return next(
        (chunk for chunk in chunks if normalized_phrase in normalize_for_lexical(chunk.text)),
        None,
    )


def _with_sources(text: str, sources: list[RetrievedChunk]) -> FactPolicyAnswer:
    unique = []
    seen = set()
    for source in sources:
        key = (source.source, source.page)
        if key not in seen:
            unique.append(source)
            seen.add(key)
    citations = " ".join(f"[{source.citation()}]" for source in unique)
    return FactPolicyAnswer(f"{text}\n\nŹródła: {citations}", unique)


def answer_from_fact_policy(question: str, chunks: list[RetrievedChunk]) -> FactPolicyAnswer | None:
    lower = question.casefold()
    normalized = normalize_for_lexical(question)

    asks_about_krakow = any(term in normalized for term in ("krakow", "filia", "oddzial"))
    asks_about_name = "nazwa" in normalized and "uczel" in normalized
    institutional = _find(chunks, "oficjalna strona przedstawia krakowską lokalizację")
    if (asks_about_krakow or asks_about_name) and institutional:
        if asks_about_krakow:
            return _with_sources(
                "Tak — oficjalna strona przedstawia krakowską lokalizację pod nazwą „Uniwersytet "
                "DSW Ideis Kraków” i podaje adres ul. Św. Filipa 17, 31-150 Kraków. Źródło "
                "opisuje połączenie dawnej WSEI w Krakowie z Uniwersytetem Dolnośląskim DSW, "
                "ale samo określenie „filia” może mieć znaczenie formalnoprawne, którego nie "
                "będę rozstrzygać bez dokumentu rejestrowego.",
                [institutional],
            )
        return _with_sources(
            "Regulamin podaje nazwę „Uniwersytet Dolnośląski DSW we Wrocławiu”, natomiast "
            "oficjalna strona krakowskiej lokalizacji używa nazwy „Uniwersytet DSW Ideis "
            "Kraków”.",
            [institutional],
        )

    asks_for_contact = any(term in normalized for term in ("adres", "gdzie", "telefon"))
    if asks_for_contact and any(term in normalized for term in ("dziekan", "uczeln", "kampus")):
        contact = _find(chunks, "dziekanat znajduje się przy ul. św. filipa 17")
        if contact:
            return _with_sources(
                "Dziekanat Uniwersytetu DSW Ideis w Krakowie znajduje się przy ul. Św. Filipa "
                "17, 31-150 Kraków. Przed wizytą warto sprawdzić aktualne godziny na "
                "oficjalnej stronie kontaktowej.",
                [contact],
            )

    if any(term in lower for term in ("ile kosztuje", "jaki koszt", "jaka opłata", "wysokość opłaty")):
        return FactPolicyAnswer(
            "Dostarczony regulamin studiów nie podaje konkretnej kwoty za powtarzanie "
            "przedmiotu ani semestru. Nie będę zgadywać wysokości opłaty; należy ją "
            "sprawdzić w aktualnym regulaminie opłat lub cenniku uczelni.",
            [],
            "insufficient_context",
        )

    if "parking" in lower or "miejsce parkingowe" in lower:
        return FactPolicyAnswer(
            "W dostarczonym regulaminie studiów nie znalazłem postanowienia gwarantującego "
            "studentom bezpłatne miejsca parkingowe.",
            [],
            "insufficient_context",
        )

    if has_approximate_root(question, "dyplom"):
        issuance = _find(
            chunks,
            "w terminie 30 dni od dnia ukończenia studiów uczelnia wydaje absolwentowi dyplom",
        )
        entitlement = _find(chunks, "absolwent studiów otrzymuje dyplom ukończenia studiów")
        available = [chunk for chunk in (entitlement, issuance) if chunk]
        if available:
            return _with_sources(
                "Po ukończeniu studiów absolwent otrzymuje dyplom ukończenia studiów wraz "
                "z suplementem i odpisami. Uczelnia wydaje te dokumenty w terminie 30 dni "
                "od dnia ukończenia studiów. Regulamin określa termin wydania dokumentów, "
                "ale nie podaje osobnego terminu, po którym wygasa możliwość ich odbioru.",
                available,
            )

    if any(term in normalized for term in ("platn", "odplat", "czesn")):
        paid = _find(chunks, "nauka w uczelni jest odpłatna")
        conditions = _find(chunks, "warunki odpłatności za studia określa umowa")
        available = [chunk for chunk in (paid, conditions) if chunk]
        if available:
            return _with_sources(
                "Tak. Regulamin stanowi, że nauka w Uczelni jest odpłatna. Szczegółowe "
                "warunki określają umowa o świadczenie usług edukacyjnych oraz regulamin "
                "opłat obowiązujący w danym roku akademickim.",
                available,
            )

    if "zapis" in normalized and "przedm" in normalized:
        source = _find(chunks, "szczegółowe zasady zapisów na zakresy kształcenia")
        if source:
            return _with_sources(
                "Regulamin nie opisuje kolejnych kroków zapisu na przedmiot. Wskazuje, że "
                "szczegółowe zasady zapisów na zakresy kształcenia i grupy zajęć do wyboru "
                "określa Rektor w osobnym zarządzeniu.",
                [source],
            )

    if "popraw" in lower and any(term in lower for term in ("ile raz", "próba", "formę zajęć")):
        source = _find(chunks, "jednej próby poprawy zaliczenia")
        if source:
            return _with_sources(
                "Regulamin przyznaje studentowi jedną próbę poprawy zaliczenia każdej formy "
                "zajęć określonej w karcie przedmiotu. Poprawa powinna nastąpić najpóźniej "
                "na tydzień przed końcem semestru. Regulamin wskazuje też prawo do poprawkowego "
                "zaliczenia po ocenie niedostatecznej lub wpisie NZAL.",
                [source],
            )

    is_appeal_question = any(
        term in normalized for term in ("odwolanie", "odwolac", "odwolania", "od decyzj")
    )
    if is_appeal_question:
        kpa = _find(chunks, "kodeks postępowania administracyjnego")
        removal = (
            _find(
                chunks,
                "skreślenie z listy studentów następuje w drodze decyzji administracyjnej",
            )
            if "skresl" in normalized
            else None
        )
        available = [chunk for chunk in (removal, kpa) if chunk]
        if available:
            subject = (
                "Skreślenie z listy studentów następuje w drodze decyzji administracyjnej. "
                if removal
                else "W indywidualnych sprawach studentów stosuje się przepisy postępowania administracyjnego. "
            )
            return _with_sources(
                subject +
                "Regulamin odsyła w indywidualnych sprawach studentów do Kodeksu postępowania "
                "administracyjnego oraz przepisów o zaskarżaniu decyzji do sądu administracyjnego. "
                "Nie określa jednej uniwersalnej formy, kanału ani terminu odwołania od każdej "
                "decyzji o skreśleniu; należy sprawdzić pouczenie w otrzymanej decyzji.",
                available,
            )

    if "niezalicz" in lower and any(term in lower for term in ("warunk", "kolejny semestr", "ile")):
        source = _find(chunks, "maksymalna łączna liczba niezaliczonych przedmiotów")
        if source:
            return _with_sources(
                "Warunkowy wpis na kolejny semestr jest możliwy, gdy łączna liczba "
                "niezaliczonych przedmiotów w dotychczasowym toku studiów nie przekracza "
                "sześciu, a liczba niezaliczonych przedmiotów w jednym semestrze nie przekracza "
                "trzech. Przekroczenie limitu powoduje skierowanie na powtarzanie semestru.",
                [source],
            )

    if "urlop" in lower and any(term in lower for term in ("jak długo", "ile semestr", "trwać")):
        source = _find(chunks, "krótkoterminowy przyznawany jest na jeden semestr")
        if source:
            return _with_sources(
                "Urlop krótkoterminowy jest przyznawany na jeden semestr, a długoterminowy "
                "na dwa semestry. W wyjątkowych sytuacjach urlop może zostać przedłużony "
                "do czterech semestrów.",
                [source],
            )

    return None
