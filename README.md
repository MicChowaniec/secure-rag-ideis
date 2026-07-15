# Bezpieczny chatbot RAG w Telegramie — temat 1

Numer indeksu `13897` daje `1+3+8+9+7=28 → 2+8=10 → 1`, dlatego chatbot jest
**asystentem regulaminu studiów**. Odpowiada wyłącznie na pytania o zasady studiowania,
zaliczenia, egzaminy i prawa studenta, korzystając z lokalnych dokumentów PDF.

## Co zawiera projekt

Przepływ wiadomości jest zgodny z założeniami zadania:

```text
Telegram → maskowanie PII → Bielik Guard → NER/słowa kluczowe
         → wyszukiwanie wektorowe → prompt z cytowanym PDF → Ollama
         → maskowanie PII i Bielik Guard odpowiedzi → Telegram + bezpieczny audyt
```

- `openai/privacy-filter` działa lokalnie jako klasyfikator tokenów z Hugging Face;
- `speakleash/Bielik-Guard-0.1B-v1.1` klasyfikuje polskie treści niebezpieczne;
- SQLite przechowuje oryginalne fragmenty, metadane stron i wektory;
- embeddingi pochodzą z Ollama (`nomic-embed-text`), a bez Ollama używany jest jawny,
  deterministyczny fallback haszujący na potrzeby demonstracji;
- przed embeddingiem pytania i dokumenty przechodzą tę samą normalizację Unicode,
  naprawę wyrazów podzielonych na końcu linii oraz ujednolicenie oznaczeń stopnia studiów;
  oryginalna treść pozostaje niezmieniona na potrzeby cytowania;
- lokalny Bielik przez Ollama generuje odpowiedź wyłącznie z dostarczonego kontekstu;
- log JSONL zapisuje wyłącznie zamaskowany input, typy PII, klasyfikację, NER, słowa
  kluczowe i identyfikatory źródeł — nigdy surowe dane wrażliwe.

> Bielik Guard ma pięć natywnych kategorii: `HATE`, `VULGAR`, `SEX`, `CRIME` i
> `SELF-HARM`. Projekt mapuje je odpowiednio na `hate_speech`, `toxic`, `sexual`,
> `violence` i `self_harm`. Etykieta `spam` pochodzi z osobnej jawnej reguły. Model nie
> powinien być opisywany jako natywnie obsługujący siedem kategorii.

## Najprostszy start na Windows

Rozpakuj paczkę i kliknij `Uruchom_Chatbot_IDEIS.exe`. Launcher:

1. wykrywa Pythona 3.11+ albo instaluje Python 3.12 dla bieżącego użytkownika;
2. tworzy `.venv` i instaluje wszystkie zależności;
3. wykrywa Ollamę i pobiera brakujące modele;
4. pozwala Ollamie automatycznie dobrać backend do sprzętu (CUDA, ROCm, Vulkan lub CPU),
   testuje prawdziwą generację, a po błędzie uruchamia izolowany fallback `cpu_avx2`;
5. pyta lokalnie o token Telegrama i opcjonalny token Hugging Face;
6. indeksuje regulamin IDEIS i pokazuje menu aplikacji oraz testów.

Nie uruchamiaj `python -m pip ...` z `C:\WINDOWS\system32`. Jeśli Windows zgłasza
„Python was not found”, użyj launchera; ręczne polecenia działają dopiero po instalacji
Pythona i przejściu do rozpakowanego folderu projektu.

## Szybki start ręczny — demonstracja bez dużych modeli

W PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m secure_rag_bot.ingest
python -m secure_rag_bot.demo
```

Tryb demonstracyjny wykonuje cały pipeline. Jeśli Ollama nie odpowiada, odpowiedź jest
ekstrakcyjnym fragmentem najlepszego źródła i jest tak wyraźnie oznaczona.

## Pełny tryb lokalny

1. Zainstaluj Ollama i pobierz modele:

```powershell
ollama pull qooba/bielik-1.5b-v3.0-instruct:Q8_0
ollama pull nomic-embed-text
```

2. Zainstaluj obsługę modeli klasyfikacyjnych:

```powershell
python -m pip install -e ".[models]"
```

3. Skopiuj `.env.example` do `.env` albo ustaw zmienne w terminalu. Projekt automatycznie
   ładuje lokalny `.env`, który jest wykluczony z archiwów wynikowych i kontroli wersji:

```powershell
$env:USE_PRIVACY_MODEL="true"
$env:USE_BIELIK_GUARD="true"
$env:HF_TOKEN="token-read-z-Hugging-Face"
$env:TELEGRAM_BOT_TOKEN="token-z-BotFather"
python -m secure_rag_bot.ingest
python -m secure_rag_bot.telegram_app
```

Ponowne indeksowanie po uruchomieniu Ollama jest ważne: baza musi używać tych samych
embeddingów przy zapisie i wyszukiwaniu.

## Konfiguracja polityk

Najważniejsze zmienne środowiskowe:

| Zmienna | Domyślnie | Znaczenie |
|---|---:|---|
| `PII_POLICY` | `mask` | `mask` kontynuuje po maskowaniu, `block` odrzuca input |
| `SAFETY_THRESHOLD` | `0.65` | próg decyzji klasyfikatora bezpieczeństwa |
| `RAG_TOP_K` | `3` | liczba różnych stron przekazanych do promptu |
| `RAG_MIN_SCORE` | `0.08` | minimalne podobieństwo wymagane do odpowiedzi |
| `USE_PRIVACY_MODEL` | `false` | włącza właściwy `openai/privacy-filter` |
| `USE_BIELIK_GUARD` | `false` | włącza właściwy model Bielik Guard |

Jeśli model jest włączony, lecz nie można go załadować, wynik audytu jawnie wskazuje
`regex-fallback` albo `heuristic-fallback`. Dzięki temu demonstracja nie udaje inferencji
modelowej.

Bielik Guard wymaga zalogowania się na Hugging Face, zaakceptowania warunków na stronie
modelu oraz tokenu z prawem `read`. Bez tego launcher pozostawia jawny fallback
heurystyczny. Tokenów Telegram ani Hugging Face nie należy wysyłać innym osobom.

## Testy

```powershell
python -m unittest discover -s tests -v
python -m secure_rag_bot.evaluation
python -m secure_rag_bot.evaluation --live --cases evals/live_factual_cases.json
```

Testy obejmują maskowanie PII, klasyfikację treści, odmowę poza domeną, prompt injection,
limity wejścia, filtr halucynacji, 40 przypadków granicznych i osobne asercje faktograficzne.
Pełna definicja granic znajduje się w `docs/BOUNDARIES_AND_TESTS.md`.

## Scenariusze do prezentacji

1. Pytanie domenowe: „Ile razy mogę zdawać egzamin poprawkowy?” — pełna ścieżka RAG,
   strony źródłowe i filtr wyjścia.
2. PII: imię, nazwisko i e-mail + pytanie o skreślenie — maskowanie przed logiem,
   klasyfikatorem, retrieverem i LLM.
3. Pytanie o pogodę lub pizzę — odmowa poza zakresem bez wywołania LLM.

Można zamienić scenariusz 3 na bezpieczną demonstrację treści ryzykownej; filtr zwróci
kontrolowany komunikat i nie uruchomi retrievera ani LLM.

## Dane i ograniczenia

Bazę tworzy oficjalny „Regulamin studiów Uniwersytetu DSW Ideis”, obowiązujący od roku
akademickiego 2025/2026. To nadal system informacyjny, a nie porada prawna. Źródło i data
dostępu znajdują się w `data/pdf/SOURCES.md`.

Na Windows najprościej uruchomić `Uruchom_Chatbot_IDEIS.exe` z głównego folderu paczki.
Launcher instaluje Pythona i zależności, sprawdza Ollamę, pobiera brakujące modele,
konfiguruje tokeny, indeksuje PDF i udostępnia menu testów oraz Telegrama. Ollama dobiera
backend do sprzętu danego komputera. Jeśli automatycznie wybrany backend nie generuje
odpowiedzi, launcher uruchamia wariant CPU na porcie 11435. Instalator projektu nigdy nie
instaluje ani nie aktualizuje sterowników karty graficznej.

Privacy Filter jest przede wszystkim modelem angielskim i producent zaleca testy
domenowe dla innych języków. Bielik Guard nie wykrywa dezinformacji ani jailbreaków.
Żaden filtr nie daje stuprocentowej skuteczności, dlatego rozwiązanie łączy model,
reguły, ograniczenie domeny, brak logowania surowego inputu i filtr wyjścia.
