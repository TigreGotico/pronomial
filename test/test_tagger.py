import unittest

from pronomial.lang.en import pos_tag_en


class TestEnglishTagger(unittest.TestCase):
    def test_pos_tag_en_returns_tags(self):
        """The English tagger must tag, or raise the missing-data error.

        nltk 3.8.2 renamed the tagger resource to
        averaged_perceptron_tagger_eng. The old fallback downloaded only the
        previous name and then called itself again, so the lookup failed on
        every pass and the caller saw RecursionError. Every test in this
        directory hit it, which is why none of them ran.
        """
        tagged = pos_tag_en("Alice invited Marcia to the store")
        self.assertTrue(tagged)
        for pair in tagged:
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
        self.assertEqual([w for w, _ in tagged][0], "Alice")


if __name__ == "__main__":
    unittest.main()
