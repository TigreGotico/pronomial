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


def strip_possessive_en(token: str) -> str:
    """Return the noun inside an English possessive token.

    quebra_frases keeps a possessive whole, so "man's" arrives as one token.
    The tagger reads that as JJ and the noun stops being a candidate
    antecedent; a name keeps its clitic and reaches the gender classifier as
    "Mary's", which matches nothing. Both want the bare noun.
    """
    for clitic in ("'s", "\u2019s", "s'", "s\u2019"):
        if len(token) > len(clitic) and token.endswith(clitic):
            return token[:-len(clitic)] + ("s" if clitic[0] == "s" else "")
    return token


def pos_tag_en(tokens):
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)

    # Tag the noun inside a possessive, and report it in place of the token.
    # Callers index by position, so the list length must not change.
    tokens = [strip_possessive_en(t) for t in tokens]

    try:
        postagged = nltk.pos_tag(tokens)
    except LookupError:
        # nltk 3.8.2 renamed the English tagger resource to
        # averaged_perceptron_tagger_eng. Downloading only the old name left
        # the lookup failing, and the fallback called itself again on every
        # call: the result was RecursionError, not a missing-data message.
        # Download both names, then retry once and let a second failure raise.
        for resource in ("averaged_perceptron_tagger_eng",
                         "averaged_perceptron_tagger"):
            nltk.download(resource, quiet=True)
        postagged = nltk.pos_tag(tokens)

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
