"""Prompt cleaner: the "jacket black" -> "change jacket to black" recolor and when it must not fire."""

from __future__ import annotations

import unittest

from backend.prompt_fix import normalize_prompt


def english(text: str, **kw) -> str:
    return normalize_prompt(text, frame=False, **kw)["english"]


class RecolorTests(unittest.TestCase):
    def test_noun_then_colour_is_a_recolor(self):
        self.assertEqual(english("jacket black"), "change jacket to black")
        self.assertEqual(english("hair brown, eyes green"), "change hair to brown, eyes green")

    def test_colour_describing_the_next_word_is_kept(self):
        for text in ("navy blue pants", "long black hair", "wearing white shirt", "light blue dress"):
            with self.subTest(text=text):
                self.assertEqual(english(text), text)

    def test_shade_words_are_not_nouns(self):
        self.assertEqual(english("pants navy blue"), "pants navy blue")

    def test_recolor_off(self):
        self.assertEqual(english("jacket black", recolor=False), "jacket black")


if __name__ == "__main__":
    unittest.main()
