import nltk
from quebra_frases import word_tokenize

PRONOUNS_EN = {
    'male': ['he', 'him', 'himself', 'his'],
    'female': ['she', 'her', 'herself', 'hers'],
    'first': ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our',
              'ours', 'ourselves'],
    'neutral': ['it', 'itself', 'its'],
    'plural': ['they', 'them', 'themselves', 'their', 'theirs', "who"]
}

WITH_EN = ["with"]
WITH_FOLLOWUP_EN = ["him", "her", "them"]

THAT_EN = ["that"]
THAT_FOLLOWUP_EN = ["he", "she", "they"]

IN_EN = ["in", "into"]
IN_FOLLOWUP_EN = ["his", "her", "their"]

PRONOUN_TAG_EN = ['PRP', 'PRP$', 'WP', 'WP$']
NOUN_TAG_EN = ['NN', 'NNP']
JJ_TAG_EN = ['JJ']
PLURAL_NOUN_TAG_EN = ['NNS', 'NNPS']
SUBJ_TAG_EN = ["nsubj", "dobj"]

NEUTRAL_WORDS_EN = ["in"]  # if word before Noun -> neutral not male nor female

NAME_JOINER_EN = " and "

GENDERED_WORDS_EN = {
    "female": ["mom", "mother", "woman", "women", "aunt", "girl", "girls",
               "sister", "sisters", "mothers"],
    "male": ["dad", "father", "man", "men", "uncle", "boy", "boys",
             "brother", "brothers", "fathers"]
}


# nltk 3.9 renamed the English tagger resource to
# "averaged_perceptron_tagger_eng". The old name is still a package that
# downloads and still reports success, but it does not satisfy the lookup that
# nltk.pos_tag makes, so a download of the old name alone never clears the
# LookupError. The old name is kept second for nltk older than 3.9, where it is
# the only one that exists.
TAGGER_RESOURCES_EN = ("averaged_perceptron_tagger_eng",
                       "averaged_perceptron_tagger")

# One download attempt per process. Without this, a tagger that stays absent
# makes every call download again, and the retry must never call back into the
# function that started it.
_tagger_download_done = False


def _nltk_pos_tag_en(tokens):
    """Tag with nltk, and download the tagger once if it is absent.

    Raises LookupError when the tagger is still absent after the download,
    which names the missing resource. It does not retry a second time.
    """
    global _tagger_download_done
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        if _tagger_download_done:
            raise
        _tagger_download_done = True
        for resource in TAGGER_RESOURCES_EN[:-1]:
            try:
                nltk.download(resource)
                return nltk.pos_tag(tokens)
            except LookupError:
                # This name is not the one this nltk looks for. Try the next.
                continue
            except Exception:
                # An unknown resource name, or no network. The last attempt
                # below reports the real problem.
                break
        try:
            nltk.download(TAGGER_RESOURCES_EN[-1])
        except Exception:
            pass
        return nltk.pos_tag(tokens)


def pos_tag_en(tokens):
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)

    postagged = _nltk_pos_tag_en(tokens)

    # HACK this fixes some know failures from postag
    # this is not sustainable but important cases can be added at any time
    # PRs + unittests welcome!
    ONOFF_VERBS = ["turn"]
    ON_OFF = ["on", "off"]
    IT_VERBS = ["change"]
    WHILE = ["while"]

    for idx, (w, t) in enumerate(postagged):
        next_w, next_t = postagged[idx + 1] if \
                             idx < len(postagged) - 1 else ("", "")

        # "turn on" falsely detected as ('Turn', 'NN'), ('on', 'IN')
        if w.lower() in ONOFF_VERBS and next_w.lower() in ON_OFF:
            # turn on/off
            if t == "NN":
                postagged[idx] = (w, "VB")
                postagged[idx + 1] = (next_w, "RP")

        # "change it" falsely detected as ('change', 'NN'), ('it', 'PRP')
        elif w.lower() in IT_VERBS and next_w.lower() == "it":
            if t == "NN":
                postagged[idx] = (w, "VB")
                postagged[idx + 1] = (next_w, "PRP")

    # END HACK

    return postagged


def is_plural_en(text):
    return text.endswith("s")
