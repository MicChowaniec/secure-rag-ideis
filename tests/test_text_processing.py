import unittest

from secure_rag_bot.rag import Embedder
from secure_rag_bot.text_processing import (
    has_approximate_root,
    clean_pdf_text,
    normalize_for_embedding,
    normalize_for_lexical,
    normalize_user_text,
)


class TextProcessingTests(unittest.TestCase):
    def test_approximate_root_accepts_one_typo_but_not_unrelated_word(self):
        self.assertTrue(has_approximate_root("odbiór duplomu", "dyplom"))
        self.assertFalse(has_approximate_root("odbiór paczki", "dyplom"))

    def test_repairs_pdf_line_hyphenation(self):
        self.assertEqual(clean_pdf_text("inży-\nnierska  praca"), "inżynierska praca")

    def test_embedding_normalization_is_shared_and_preserves_polish(self):
        normalized = normalize_for_embedding("  STUDIA II stopnia\n")
        self.assertEqual(normalized, "studia drugiego stopnia")
        self.assertIn("ó", normalize_for_embedding("studiów"))

    def test_lexical_normalization_handles_diacritics(self):
        self.assertEqual(normalize_for_lexical("płatność i odwołanie"), "platnosc i odwolanie")
        self.assertEqual(normalize_for_lexical("INŻYNIERSKIE"), "inzynierskie")

    def test_user_normalization_removes_controls(self):
        self.assertEqual(normalize_user_text("  test\x00\x01  "), "test")

    def test_hash_embedding_uses_the_same_normalized_representation(self):
        embedder = Embedder("http://127.0.0.1:1", "none")
        self.assertEqual(
            embedder.hash_embed("STUDIA II stopnia"),
            embedder.hash_embed("studia drugiego stopnia"),
        )


if __name__ == "__main__":
    unittest.main()
