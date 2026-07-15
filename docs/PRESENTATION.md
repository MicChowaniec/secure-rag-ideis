# Plan prezentacji (około 8–10 minut)

## 1. Temat

Pokaż redukcję indeksu: `13897 → 28 → 10 → 1`. Temat: asystent regulaminu studiów.

## 2. Architektura

Krótko przejdź przez diagram z README. Podkreśl, że żaden komponent po filtrze PII nie
widzi surowego adresu e-mail, a model LLM nie jest wywoływany dla treści niebezpiecznej
lub pytania poza domeną.

## 3. Trzy przebiegi

Uruchom `Uruchom_Chatbot_IDEIS.exe` i wybierz scenariusze demonstracyjne albo wykonaj
`python -m secure_rag_bot.demo`:

- przebieg 1: czyste pytanie domenowe, wynik `answered`, fragmenty i strony PDF;
- przebieg 2: PII zostaje zamienione na `[OSOBA_PRYWATNA]` i `[EMAIL]`, a log zawiera
  jedynie typy danych;
- przebieg 3: pytanie o pizzę/pogodę kończy się `out_of_scope`.

Terminal pokazuje automatycznie wszystkie dziewięć etapów pipeline'u dla każdej
wiadomości, w tym backendy modeli, confidence score, encje, słowa kluczowe, źródła i
czasy wykonania. W przebiegu z PII wejście jest już zamaskowane. Na końcu można dodatkowo
pokazać ostatnie trzy rekordy `data/audit.jsonl`. Nie pokazuj prawdziwego tokena Telegrama
ani prawdziwych danych osobowych.

## 4. Rzetelność techniczna

Wyjaśnij różnicę między właściwymi modelami a fallbackami. Wskaż rzeczywistą taksonomię
Bielik Guard oraz osobną regułę spamu. Zaznacz, że cytowanie strony ogranicza, ale nie
eliminuje halucynacji — dlatego prompt pozwala odpowiedzieć „brak informacji”.

## 5. Testy i ograniczenia

Pokaż zielony wynik testów. Wymień ograniczenia językowe Privacy Filter, brak gwarancji
100% wykrycia PII i konieczność użycia aktualnego regulaminu właściwej uczelni.
