from os.path import join, dirname, isfile
import csv
import nltk
import pickle

from nltk.tokenize import TreebankWordTokenizer
from quebra_frases import sentence_tokenize
from quebra_frases import word_tokenize as _word_tokenize_default
from pronomial.lang.pt import pos_tag_pt
from pronomial.lang.en import pos_tag_en, is_plural_en
from pronomial.lang.es import pos_tag_es
from pronomial.lang.ca import pos_tag_ca


#: The English tagger is nltk's perceptron model, trained on the Penn
#: Treebank, so English has to be tokenized the way the Treebank is: a
#: possessive or a contraction is its own token (``man`` + ``'s``,
#: ``do`` + ``n't``). The default tokenizer keeps an apostrophe inside the
#: word, which is right for Catalan ``l'ha`` and Portuguese ``d'agua`` and
#: wrong here: with ``man's`` as one token the tagger never sees ``man``, so no
#: pronoun can resolve to it.
_TREEBANK = TreebankWordTokenizer()


def _word_tokenize_en(text):
    """Treebank tokens for English. The Treebank tokenizer splits a final
    period only at the end of its input, so the text is split into sentences
    first; otherwise ``friend.`` stays one token in the middle of a text."""
    tokens = []
    for sentence in sentence_tokenize(text):
        tokens.extend(_TREEBANK.tokenize(sentence))
    return tokens


def word_tokenize(text, lang="en"):
    """Tokens for *lang*, matching what that language's tagger was trained on.

    Every caller that indexes into this list -- ``detect_nouns``,
    ``score_corefs``, ``solve_corefs``, ``replace_corefs`` -- must pass the
    same lang as the tagger, or the indices name different words on each side.
    """
    if lang.startswith("en"):
        return _word_tokenize_en(text)
    return _word_tokenize_default(text)


def pos_tag(text, lang="en"):
    if lang.startswith("en"):
        return pos_tag_en(text)
    if lang.startswith("pt"):
        return pos_tag_pt(text)
    if lang.startswith("es"):
        return pos_tag_es(text)
    if lang.startswith("pt"):
        return pos_tag_ca(text)
    raise NotImplementedError


def _get_features(name):
    return {
        'suffix1': name[-1],
        'suffix2': name[-2:],
        'suffix3': name[-3:],
        'suffix4': name[-4:],
        'suffix5': name[-5:]
    }


def train_gender_classifier(path):
    dataset = join(dirname(__file__), "res", "names.csv")
    with open(dataset) as f:
        names = [tuple(line) for line in csv.reader(f)]
    train_data = [(_get_features(n), g) for (n, g) in names]
    clf = nltk.NaiveBayesClassifier.train(train_data)
    path = path or join(dirname(__file__), "res", "name_gender.pkl")
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    return clf


def load_gender_classifier(path=None):
    path = path or join(dirname(__file__), "res", "name_gender.pkl")
    if not isfile(path):
        train_gender_classifier(path)
    with open(path, "rb") as f:
        tagger = pickle.load(f)
    return tagger


GENDER = load_gender_classifier()


def predict_gender(word, text="", lang="en"):
    gender = None
    if lang.startswith("en"):
        from pronomial.lang.en import GENDERED_WORDS_EN
        for k, v in GENDERED_WORDS_EN.items():
            if word.lower() in v:
                gender = k
                break
    if lang.startswith("pt"):
        from pronomial.lang.pt import predict_gender_pt
        gender = predict_gender_pt(word, text)
    if lang.startswith("es"):
        from pronomial.lang.es import predict_gender_es
        gender = predict_gender_es(word, text)
    if lang.startswith("ca"):
        from pronomial.lang.ca import predict_gender_ca
        gender = predict_gender_ca(word, text)

    return gender or GENDER.classify(_get_features(word))


def is_plural(text, lang="en"):
    if lang.startswith("en"):
        return is_plural_en(text)
    return text.endswith("s")
