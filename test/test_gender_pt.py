import unittest

from pronomial.lang.pt import predict_gender_pt


class TestPortugueseGender(unittest.TestCase):
    def test_plural_of_a_listed_feminine_noun(self):
        # "mães" and "irmãs" are the plurals of listed feminine nouns, and
        # neither ends in "a" or "as", so only the list can decide them.
        self.assertEqual(predict_gender_pt("mães"), "female")
        self.assertEqual(predict_gender_pt("irmãs"), "female")

    def test_plural_of_a_listed_masculine_noun(self):
        self.assertEqual(predict_gender_pt("pais"), "male")
        self.assertEqual(predict_gender_pt("tios"), "male")

    def test_feminine_demonstratives(self):
        # "flor" is feminine and has no gendered ending; the determiner decides.
        self.assertEqual(predict_gender_pt("flor", "esta flor"), "female")
        self.assertEqual(predict_gender_pt("flores", "estas flores"), "female")
        self.assertEqual(predict_gender_pt("flor", "essa flor"), "female")

    def test_masculine_demonstratives(self):
        self.assertEqual(predict_gender_pt("amor", "este amor"), "male")
        self.assertEqual(predict_gender_pt("amor", "esse amor"), "male")


if __name__ == "__main__":
    unittest.main()
