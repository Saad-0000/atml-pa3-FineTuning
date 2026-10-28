"""Tests for data normalization, leakage filtering, and length cap rules."""

import unittest
from task1.data import normalize_text


class TestDataFiltering(unittest.TestCase):
    def test_text_normalization(self):
        raw = "  Solve  for   x:\n\n“x + 5 = 10” — What’s x?  "
        norm = normalize_text(raw)
        # Check lowercased, whitespace collapsed, curly quotes converted
        self.assertEqual(norm, 'solve for x: "x + 5 = 10" - what\'s x?')

    def test_unicode_nfkc_normalization(self):
        # Half-width vs full-width chars
        full_width = "１２３ ａｂｃ"
        norm = normalize_text(full_width)
        self.assertEqual(norm, "123 abc")

    def test_leakage_detection(self):
        gsm8k_set = {normalize_text("Natalia sold clips to 48 of her friends in April.")}
        match_query = "natalia  sold clips to 48  of her friends in april. "
        non_match_query = "Janet's ducks lay 16 eggs per day."

        self.assertIn(normalize_text(match_query), gsm8k_set)
        self.assertNotIn(normalize_text(non_match_query), gsm8k_set)


if __name__ == "__main__":
    unittest.main()
