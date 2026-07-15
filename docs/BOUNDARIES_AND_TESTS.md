# Granice systemu i strategia testów

## Zdefiniowane granice

### 1. Granica tematyczna

Bot odpowiada tylko o regulaminie studiów IDEIS: prawach i obowiązkach studenta,
zaliczeniach, egzaminach, ECTS, urlopach, skreśleniach, wznowieniach, przeniesieniach,
opłatach i ukończeniu studiów. Pogoda, przepisy, polityka, programowanie i porady medyczne
kończą się statusem `out_of_scope` bez wywołania LLM.

### 2. Granica wiedzy

Odpowiedź musi opierać się na jednym z 58 fragmentów oficjalnego PDF. Minimalny wynik
retrievalu wynosi `RAG_MIN_SCORE`. Brak podstawy daje `insufficient_context`.

Generator nie może dopowiadać terminów, formularzy, adresów, sposobów złożenia pisma ani
liczb nieobecnych w kontekście. Filtr ugruntowania sprawdza liczby, słownictwo proceduralne,
pokrycie treści i długość. Podejrzana odpowiedź jest zastępowana cytatem źródłowym ze
statusem `grounding_fallback`.

### 3. Granica danych wejściowych

- pusty tekst: `invalid_input`;
- maksimum 4000 znaków;
- normalizacja Unicode NFKC i usunięcie znaków sterujących;
- Telegram przyjmuje wyłącznie wiadomości tekstowe;
- dane PII są maskowane przed klasyfikacją, NER, RAG, LLM i audytem.

### 4. Granica bezpieczeństwa

Bielik Guard obsługuje `HATE`, `VULGAR`, `SEX`, `CRIME` i `SELF-HARM`; spam ma osobną
regułę. Prompt injection i próby ujawnienia instrukcji systemowej są blokowane osobnymi
wzorcami, ponieważ Bielik Guard jawnie nie jest modelem do wykrywania jailbreaków.

### 5. Granica dostępności

Brak Ollamy nie powoduje wyjątku dla użytkownika — dostępny jest jawny fallback
ekstrakcyjny. Błąd CUDA jest wykrywany krótką próbą generacji; launcher uruchamia wtedy
`OLLAMA_LLM_LIBRARY=cpu_avx2` na osobnym porcie. Nieudane załadowanie modelu klasyfikacji
jest widoczne jako `*-fallback`, a nie ukrywane.

## Zestaw testowy

Plik `evals/cases.json` zawiera 40 przypadków:

- 8 pytań domenowych;
- 5 pytań poza zakresem;
- 5 wariantów PII;
- 6 kategorii bezpieczeństwa, w tym spam;
- 4 próby prompt injection;
- 8 przypadków brzegowych: pusty tekst, whitespace, przekroczenie limitu, emoji, wielkie
  litery, literówki, brak informacji i znaki sterujące.
- 4 regresje ugruntowania czasu studiów: pytanie ogólne, magisterskie oraz dwa warianty
  pytania o studia inżynierskie bez informacji źródłowej.

Runner kontroluje status, źródła, długość odpowiedzi, wycieki PII, artefakty wyjątków i
backend generacji. Tryb `--live` używa rzeczywistego modelu Ollama.

```powershell
python -m unittest discover -s tests -v
python -m secure_rag_bot.evaluation
python -m secure_rag_bot.evaluation --live
```

## Testy faktograficzne do ręcznej oceny

1. „Ile razy mogę poprawić zaliczenie?” — oczekiwany zapis z §19 ust. 15: jedna próba
   poprawy każdej formy zajęć; nie wolno rozszerzać tego automatycznie na procedury, których
   dokument nie opisuje.
2. „Jak odwołać się od skreślenia?” — regulamin mówi o decyzji administracyjnej (§26) i
   stosowaniu KPA (§34), ale nie opisuje formularza ani kanału wysyłki; bot nie może ich
   wymyślić.
3. „Ile niezaliczonych przedmiotów dopuszcza wpis warunkowy?” — maksymalnie sześć w toku
   studiów i trzy w jednym semestrze (§22 ust. 3).
4. „Jak długo może trwać urlop?” — jeden lub dwa semestry, wyjątkowo do czterech (§25).
5. „Czy darmowy parking jest gwarantowany?” — brak takiej informacji; bot powinien odmówić
   potwierdzenia zamiast zgadywać.
