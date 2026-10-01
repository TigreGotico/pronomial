"""The English tagger download must not recurse.

nltk 3.9 renamed the English tagger resource to
"averaged_perceptron_tagger_eng". pronomial 0.0.8 downloaded the old name,
which still reports success without satisfying the lookup, and then called
pos_tag_en again. On a machine with no tagger that repeated until Python
raised RecursionError, so every English call failed.
"""
import unittest
from unittest.mock import patch

import nltk

import pronomial.lang.en as en


class MissingTagger(Exception):
    """Marks the fake nltk.pos_tag calls, which never find a tagger."""


def _absent_tagger(tokens):
    raise LookupError("Resource 'averaged_perceptron_tagger_eng' not found.")


class TestEnTaggerResource(unittest.TestCase):

    _ABSENT = object()

    def setUp(self):
        # The flag is module state, and one test must not change what the
        # next test sees. getattr with a default keeps this file a real
        # oracle: against a pronomial that has no flag at all, each test
        # still reaches its own assertion and fails for its own reason,
        # instead of every test erroring here in setUp.
        self._saved = getattr(en, "_tagger_download_done", self._ABSENT)
        en._tagger_download_done = False

    def tearDown(self):
        if self._saved is self._ABSENT:
            del en._tagger_download_done
        else:
            en._tagger_download_done = self._saved

    def test_the_new_resource_name_is_tried_first(self):
        self.assertEqual(en.TAGGER_RESOURCES_EN[0],
                         "averaged_perceptron_tagger_eng")
        # The old name stays, for nltk older than 3.9.
        self.assertIn("averaged_perceptron_tagger", en.TAGGER_RESOURCES_EN)

    def test_a_tagger_that_stays_absent_raises_lookuperror(self):
        """The failure a user sees names the resource; it is not a
        RecursionError."""
        downloads = []
        with patch.object(nltk, "pos_tag", _absent_tagger), \
                patch.object(nltk, "download", lambda name: downloads.append(name)):
            with self.assertRaises(LookupError):
                en.pos_tag_en("he turned it on")
        # Every configured name was tried once, and no name twice.
        self.assertEqual(downloads, list(en.TAGGER_RESOURCES_EN))

    def test_a_second_call_does_not_download_again(self):
        downloads = []
        with patch.object(nltk, "pos_tag", _absent_tagger), \
                patch.object(nltk, "download", lambda name: downloads.append(name)):
            for _ in range(3):
                with self.assertRaises(LookupError):
                    en.pos_tag_en("he turned it on")
        self.assertEqual(downloads, list(en.TAGGER_RESOURCES_EN))

    def test_the_download_is_not_recursive(self):
        """A low recursion limit must not change the outcome.

        pronomial 0.0.8 fails this: it recurses once per retry, so a limit
        this low turns the LookupError into a RecursionError.
        """
        import sys
        limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(80)
            with patch.object(nltk, "pos_tag", _absent_tagger), \
                    patch.object(nltk, "download", lambda name: None):
                with self.assertRaises(LookupError):
                    en.pos_tag_en("he turned it on")
        finally:
            sys.setrecursionlimit(limit)

    def test_the_first_name_that_works_stops_the_search(self):
        """The old name is not downloaded when the new one satisfies the
        lookup."""
        downloads = []
        state = {"ok": False}

        def fake_pos_tag(tokens):
            if not state["ok"]:
                raise LookupError("Resource 'x' not found.")
            return [(t, "PRP") for t in tokens]

        def fake_download(name):
            downloads.append(name)
            if name == "averaged_perceptron_tagger_eng":
                state["ok"] = True

        with patch.object(nltk, "pos_tag", fake_pos_tag), \
                patch.object(nltk, "download", fake_download):
            tagged = en.pos_tag_en("he turned it on")

        self.assertEqual(downloads, ["averaged_perceptron_tagger_eng"])
        self.assertTrue(tagged)


if __name__ == "__main__":
    unittest.main()
