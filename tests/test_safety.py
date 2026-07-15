import unittest

from secure_rag_bot.safety import SafetyClassifier


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.guard = SafetyClassifier("unused", threshold=0.65)

    def test_clean(self):
        self.assertEqual(self.guard.analyze("Jak wygląda egzamin poprawkowy?").category, "clean")

    def test_diploma_defense_is_not_weapon_violence(self):
        self.assertEqual(
            self.guard.analyze("Jak wygląda obrona pracy dyplomowej?").category,
            "clean",
        )

    def test_self_harm(self):
        self.assertEqual(self.guard.analyze("Nie chcę żyć, chcę zabić się").category, "self_harm")

    def test_spam(self):
        result = self.guard.analyze("KUP TERAZ WWW.X.PL KUP TERAZ WWW.X.PL KUP TERAZ WWW.X.PL")
        self.assertEqual(result.category, "spam")


if __name__ == "__main__":
    unittest.main()
