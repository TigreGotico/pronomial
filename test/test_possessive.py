"""The possessive arrives as one token, and the noun inside it is a mention.

quebra_frases keeps "man's" whole. Before this, the tagger read that token as
JJ so the noun stopped being a candidate antecedent, a name kept its clitic
and reached the gender classifier as "Mary's" (matching nothing), and an
antecedent taken from such a token was substituted with the clitic attached.
"""
import unittest

from pronomial import replace_corefs, word_tokenize
from pronomial.lang.en import pos_tag_en, strip_possessive_en


class TestPossessiveClitic(unittest.TestCase):
    def test_strip_only_touches_a_possessive(self):
        self.assertEqual(strip_possessive_en("man's"), "man")
        self.assertEqual(strip_possessive_en("Mary's"), "Mary")
        self.assertEqual(strip_possessive_en("dogs'"), "dogs")
        # not possessives
        self.assertEqual(strip_possessive_en("man"), "man")
        self.assertEqual(strip_possessive_en("its"), "its")
        self.assertEqual(strip_possessive_en("'s"), "'s")
        self.assertEqual(strip_possessive_en(""), "")

    def test_the_noun_in_a_possessive_is_tagged_as_a_noun(self):
        """"man's" alone tags JJ; the noun inside it tags NN."""
        tags = dict(pos_tag_en("dog is man's best friend"))
        self.assertIn("man", tags)
        self.assertTrue(tags["man"].startswith("NN"), tags)

    def test_the_token_count_is_unchanged(self):
        """Callers index by position, so tagging must not split a token."""
        sentence = "Mary's car broke down"
        self.assertEqual(len(pos_tag_en(sentence)),
                         len(word_tokenize(sentence)))

    def test_a_possessor_can_be_the_antecedent(self):
        self.assertEqual(replace_corefs("John's mother loves him."),
                         "John's mother loves John .")

    def test_the_possessor_keeps_its_gender(self):
        """"Alice's" matched no gendered word, so "she" took "brother"."""
        self.assertEqual(replace_corefs("Alice's brother said she was late."),
                         "Alice's brother said Alice was late .")
        self.assertEqual(
            replace_corefs("Mary's car broke down. She had to walk."),
            "Mary's car broke down . Mary had to walk .")

    def test_the_substitution_drops_the_clitic(self):
        """An antecedent is inserted as the noun, never as the possessive."""
        out = replace_corefs("John's mother loves him.")
        self.assertNotIn("loves John's", out)


if __name__ == "__main__":
    unittest.main()
