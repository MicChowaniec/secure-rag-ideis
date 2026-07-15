import unittest

from secure_rag_bot.privacy import PrivacyFilter


class PrivacyFilterTests(unittest.TestCase):
    def test_masks_email_phone_and_secret(self):
        result = PrivacyFilter("unused").analyze(
            "Kontakt: ala@example.com, +48 501 502 503, hasło: Tajne123!"
        )
        self.assertNotIn("ala@example.com", result.masked_text)
        self.assertNotIn("501 502 503", result.masked_text)
        self.assertNotIn("Tajne123", result.masked_text)
        self.assertTrue(result.contains_pii)

    def test_no_pii(self):
        result = PrivacyFilter("unused").analyze("Kiedy jest egzamin poprawkowy?")
        self.assertFalse(result.contains_pii)


if __name__ == "__main__":
    unittest.main()

