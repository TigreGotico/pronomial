"""The English tagger's model lookup, and the retry that must not recurse.

`pos_tag_en` tags with nltk's perceptron model and fetches it when it is
absent. nltk 3.9 renamed that model to `averaged_perceptron_tagger_eng`, and
the old code answered a `LookupError` by downloading the OLD name and calling
itself: the download never satisfied the lookup, so every English call ended in
`RecursionError: maximum recursion depth exceeded`, with a network download
attempted at each of the thousand levels.

The first test is the one that matters, and it holds with no model on disk and
no network: whatever happens to the download, one retry is made and the error
that reaches the caller says what is missing.
"""
import unittest
from unittest.mock import patch

import nltk

from pronomial.lang import en
from pronomial.lang.en import pos_tag_en


class TestEnTaggerRetry(unittest.TestCase):

    def test_a_model_that_never_arrives_raises_lookup_error_not_recursion(self):
        """The regression guard. nltk.pos_tag always fails and the download
        reports success, which is exactly the nltk 3.9 shape: the old code
        recursed here until the stack ran out."""
        calls = []

        def always_missing(tokens):
            calls.append(tokens)
            raise LookupError("Resource averaged_perceptron_tagger_eng not found")

        with patch.object(nltk, "pos_tag", side_effect=always_missing), \
                patch.object(en, "_download_en_tagger", return_value=True) as dl:
            with self.assertRaises(LookupError) as caught:
                pos_tag_en("Mary read it")
        self.assertIn("averaged_perceptron_tagger_eng", str(caught.exception),
                      "the error must name the resource the caller has to fetch")
        self.assertEqual(len(calls), 2, "expected one tag and exactly one retry")
        self.assertEqual(dl.call_count, 1, "expected exactly one download")

    def test_the_new_model_name_is_tried_first(self):
        """nltk 3.9 and later answer only to the _eng name, so it must come
        first: the plain name downloads successfully on 3.9 and still leaves the
        lookup unsatisfied, which is what produced the recursion."""
        self.assertEqual(en._EN_TAGGER_MODELS[0],
                         "averaged_perceptron_tagger_eng")
        self.assertIn("averaged_perceptron_tagger", en._EN_TAGGER_MODELS)

    def test_a_download_that_raises_is_not_fatal_here(self):
        """No network is not this function's error to report. It must fall
        through to the tagging call, whose LookupError names the resource."""
        with patch.object(nltk, "download", side_effect=OSError("no network")):
            self.assertFalse(en._download_en_tagger())


class TestEnTagging(unittest.TestCase):
    """English tagging itself. Needs the model, which the library fetches.

    These do not skip when the model is absent: the library is meant to fetch
    it, so a failure here is the library failing, and a suite that skips its own
    oracle reports the same green as one that checked it.
    """

    @classmethod
    def setUpClass(cls):
        # Tag once so the model is on disk for the tests below, including the
        # one that asserts no download happens when it is already there.
        pos_tag_en("Mary read it")

    def test_no_download_when_the_model_is_already_there(self):
        with patch.object(en, "_download_en_tagger") as dl:
            pos_tag_en("Mary read it")
        dl.assert_not_called()

    def test_every_token_gets_a_tag(self):
        tagged = pos_tag_en("Mary bought a book and she read it")
        self.assertEqual([w for w, _ in tagged],
                         ["Mary", "bought", "a", "book", "and", "she",
                          "read", "it"])
        for word, tag in tagged:
            self.assertTrue(tag, f"{word} got an empty tag")

    def test_a_pronoun_and_a_name_are_not_tagged_alike(self):
        tags = dict(pos_tag_en("Mary bought a book and she read it"))
        self.assertNotEqual(tags["Mary"], tags["she"])


if __name__ == "__main__":
    unittest.main()
