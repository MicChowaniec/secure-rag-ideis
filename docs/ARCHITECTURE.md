# Architektura i decyzje bezpieczeństwa

## Kolejność przetwarzania

1. Wiadomość Telegram jest przyjmowana jako tekst i otrzymuje losowy `trace_id`.
2. Privacy Filter wykrywa spany PII; reguły deterministyczne chronią typowe formaty
   również wtedy, gdy model jest niedostępny.
3. Dalsze komponenty widzą tylko tekst zamaskowany.
4. Bielik Guard ocenia wejście. Spam jest osobną regułą, ponieważ nie należy do
   taksonomii Bielik Guard.
5. NER, słowa kluczowe i kategoria pytania są liczone na tekście zamaskowanym.
6. Pytanie i fragmenty PDF mają wspólną reprezentację znormalizowaną do embeddingów;
   oryginalny tekst pozostaje oddzielnie jako materiał cytowany.
7. Bramka domenowa i minimalne podobieństwo RAG odrzucają pytania obce tematycznie.
8. Dla pytań ilościowych system sprawdza, czy podmiot, miara i liczba występują w jednym
   zdaniu lub punkcie źródła oraz zachowuje kwalifikatory, np. „jednolite”.
9. Prompt zawiera tylko najlepsze fragmenty, strony, zasady cytowania i zakaz wykonywania
   instrukcji znalezionych w niezaufanym PDF.
10. Dla faktów szczególnie podatnych na halucynacje działa deterministyczna polityka
   oparta na konkretnych paragrafach; pozostałe odpowiedzi generuje Ollama.
11. Filtr ugruntowania odrzuca liczby, procedury i rozwlekłe twierdzenia niepoparte kontekstem.
12. Każda odpowiedź, również deterministyczna i ekstrakcyjna, przechodzi drugi Privacy
    Filter i Bielik Guard.
13. Audyt JSONL zapisuje metadane, ale nie zapisuje surowego inputu ani treści spanów PII.

## Granice zaufania

- Telegram i tekst użytkownika są niezaufane.
- PDF jest źródłem faktów, ale także pozostaje niezaufany jako nośnik instrukcji.
- Odpowiedź LLM jest niezaufana do chwili przejścia filtrów wyjściowych.
- Token Telegrama istnieje wyłącznie w zmiennej środowiskowej i nie trafia do promptu.

## Komponenty i fallbacki

| Etap | Pełny backend | Fallback demonstracyjny |
|---|---|---|
| PII | `openai/privacy-filter` | regex |
| Bezpieczeństwo | Bielik Guard 0.1B | jawne słowniki + reguła spamu |
| NER | reguły dla dat i organizacji | ten sam mechanizm |
| Embedding | `nomic-embed-text` w Ollama | 384-wymiarowy hash tokenów |
| Generacja | Bielik 1.5B w Ollama | cytat z najlepszego fragmentu |

Na badanym komputerze autodetekcja CUDA kończyła proces runnera błędem niezgodnego PTX.
Launcher wykonuje próbę generacji, a następnie uruchamia oficjalny backend `cpu_avx2` na
porcie 11435, jeśli standardowy serwer nie potrafi wygenerować odpowiedzi.

Fallbacki umożliwiają test i prezentację sterowania przepływem. Nie są przedstawiane
jako równoważne jakościowo właściwym modelom.
