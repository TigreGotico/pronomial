"""Which tagger pos_tag reaches for a given lang code.

The dispatch is four `lang.startswith` branches, and a copied branch is
invisible at a glance: the fourth one tested "pt" where it meant "ca", so
pos_tag_ca could not be reached and every Catalan call raised
NotImplementedError while the Catalan tagger sat loaded and unused.

These tests assert the tagger each code reaches, not the tags it produces.
English is left out on purpose: pos_tag_en recurses on nltk 3.9 and later, a
separate defect filed as T-5385, and importing it here would hang this file.
"""
import unittest

from pronomial.lang.ca import pos_tag_ca
from pronomial.lang.es import pos_tag_es
from pronomial.lang.pt import pos_tag_pt
from pronomial.utils import pos_tag

#: One sentence, spelled so that each tagger has something to disagree about.
#: "ha comprat" is Catalan, "comprou" is its Portuguese form, and the three
#: taggers are trained on different corpora.
SENTENCE = "La Maria ha comprat un llibre"


class TestPosTagDispatch(unittest.TestCase):

    def test_ca_reaches_the_catalan_tagger(self):
        self.assertEqual(pos_tag(SENTENCE, lang="ca"), pos_tag_ca(SENTENCE))

    def test_ca_with_a_region_reaches_it_too(self):
        self.assertEqual(pos_tag(SENTENCE, lang="ca-ES"), pos_tag_ca(SENTENCE))

    def test_pt_reaches_the_portuguese_tagger(self):
        self.assertEqual(pos_tag(SENTENCE, lang="pt"), pos_tag_pt(SENTENCE))
        self.assertEqual(pos_tag(SENTENCE, lang="pt-BR"), pos_tag_pt(SENTENCE))

    def test_es_reaches_the_spanish_tagger(self):
        self.assertEqual(pos_tag(SENTENCE, lang="es"), pos_tag_es(SENTENCE))

    def test_ca_and_pt_do_not_return_the_same_tags(self):
        """The failure this guards: a Catalan caller served by the Portuguese
        tagger. If the two ever agree on this sentence the test is no longer
        an oracle, so it says which words differ when it fails."""
        ca = pos_tag(SENTENCE, lang="ca")
        pt = pos_tag(SENTENCE, lang="pt")
        self.assertNotEqual(
            ca, pt,
            "ca and pt tagged identically; this sentence no longer separates "
            "the two taggers and the test needs a new one")

    def test_ca_and_es_do_not_return_the_same_tags(self):
        self.assertNotEqual(pos_tag(SENTENCE, lang="ca"),
                            pos_tag(SENTENCE, lang="es"))

    def test_an_unwired_lang_is_refused(self):
        """Every code that reaches no tagger must say so. de has no tagger
        here, and the branch order must not let it fall into one."""
        for lang in ["de", "fr", "nl-NL"]:
            with self.assertRaises(NotImplementedError):
                pos_tag(SENTENCE, lang=lang)


if __name__ == "__main__":
    unittest.main()
