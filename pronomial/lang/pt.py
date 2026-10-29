import pickle
from os.path import isfile, join, dirname
import nltk.tag
import nltk
from string import punctuation
from random import shuffle
from quebra_frases import word_tokenize

PRONOUNS_PT = {
    'male': ['ele', "lo", "dele", "nele", "seu", "eles", "seus", "deles",
             "neles"],
    'female': ['ela', "la", "dela", "nela", "sua", "elas", "suas", "delas",
               "nelas"],
    'first': ['eu', 'me', 'mim', 'nós', "comigo", "meu", "minha", "meus",
              "minhas"],
    'neutral': ["tu", "te", "ti", "lhe", "contigo", "consigo", "si"],
    'plural': ['eles', 'elas', "vós", "vocês", "lhes", "los", "las",
               "neles", "nelas", "convosco", "conosco", "connosco", "teus",
               "tuas", "seus", "suas", "nossos", "vossos", "nossas", "vossas"]
}

# special cases, word2gender mappings
GENDERED_WORDS_PT = {
    "female": ["mãe", "irmã", "tia", "amiga", "prima", "namorada", "mulher",
               "mulheres", "rapariga", "raparigas", "gaja", "gajas", "moça",
               "moças", "elas", "suas"],
    "male": ["pai", "irmão", "tio", "amigo", "primo", "namorado", "homem",
             "homens", "rapaz", "rapazes", "gajo", "gajos", "moço", "moços",
             "eles", "seus"]
}

# context rules for gender
MALE_DETERMINANTS_PT = ["o", "os", "este", "estes", "esse", "esses"]
FEMALE_DETERMINANTS_PT = ["a", "as", "esta", "estas", "essa", "essas"]


def train_pt_tagger(path):
    nltk.download('mac_morpho')
    nltk.download('floresta')

    def convert_to_universal_tag(t, reverse=False):
        tagdict = {
            'n': "NOUN",
            'num': "NUM",
            'v-fin': "VERB",
            'v-inf': "VERB",
            'v-ger': "VERB",
            'v-pcp': "VERB",
            'pron-det': "PRON",
            'pron-indp': "PRON",
            'pron-pers': "PRON",
            'art': "DET",
            'adv': "ADV",
            'conj-s': "CONJ",
            'conj-c': "CONJ",
            'conj-p': "CONJ",
            'adj': "ADJ",
            'ec': "PRT",
            'pp': "ADP",
            'prp': "ADP",
            'prop': "NOUN",
            'pro-ks-rel': "PRON",
            'proadj': "PRON",
            'prep': "ADP",
            'nprop': "NOUN",
            'vaux': "VERB",
            'propess': "PRON",
            'v': "VERB",
            'vp': "VERB",
            'in': "X",
            'prp-': "ADP",
            'adv-ks': "ADV",
            'dad': "NUM",
            'prosub': "PRON",
            'tel': "NUM",
            'ap': "NUM",
            'est': "NOUN",
            'cur': "X",
            'pcp': "VERB",
            'pro-ks': "PRON",
            'hor': "NUM",
            'pden': "ADV",
            'dat': "NUM",
            'kc': "ADP",
            'ks': "ADP",
            'adv-ks-rel': "ADV",
            'npro': "NOUN",
        }
        if t in ["N|AP", "N|DAD", "N|DAT", "N|HOR", "N|TEL"]:
            t = "NUM"
        if reverse:
            if "|" in t: t = t.split("|")[0]
        else:
            if "+" in t: t = t.split("+")[1]
            if "|" in t: t = t.split("|")[1]
            if "#" in t: t = t.split("#")[0]
        t = t.lower()
        return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)

    floresta = [[(w, convert_to_universal_tag(t))
                 for (w, t) in sent]
                for sent in nltk.corpus.floresta.tagged_sents()]
    shuffle(floresta)

    mac_morpho = [[w[0] for w in sent] for sent in
                  nltk.corpus.mac_morpho.tagged_paras()]
    mac_morpho = [
        [(w, convert_to_universal_tag(t, reverse=True))
         for (w, t) in sent] for sent in mac_morpho]
    shuffle(mac_morpho)

    regex_patterns = [
        (r"^[nN][ao]s?$", "ADP"),
        (r"^[dD][ao]s?$", "ADP"),
        (r"^[pP]el[ao]s?$", "ADP"),
        (r"^[nN]est[ae]s?$", "ADP"),
        (r"^[nN]um$", "ADP"),
        (r"^[nN]ess[ae]s?$", "ADP"),
        (r"^[nN]aquel[ae]s?$", "ADP"),
        (r"^\xe0$", "ADP"),
    ]

    def_tagger = nltk.DefaultTagger('NOUN')
    affix_tagger = nltk.AffixTagger(
        mac_morpho + floresta, backoff=def_tagger
    )
    unitagger = nltk.UnigramTagger(
        mac_morpho + floresta, backoff=affix_tagger
    )
    rx_tagger = nltk.RegexpTagger(
        regex_patterns, backoff=unitagger
    )
    tagger = nltk.BigramTagger(
        floresta, backoff=rx_tagger
    )
    tagger = nltk.BrillTaggerTrainer(tagger, nltk.brill.fntbl37())
    tagger = tagger.train(floresta, max_rules=100)

    with open(path, "wb") as f:
        pickle.dump(tagger, f)

    return tagger


def load_pt_tagger(path=None):
    path = path or join(dirname(dirname(__file__)), "res", "pt_tagger.pkl")
    if not isfile(path):
        train_pt_tagger(path)
    with open(path, "rb") as f:
        tagger = pickle.load(f)
    return tagger


_POSTAGGER = load_pt_tagger()


def pos_tag_pt(tokens):
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)
    postagged = _POSTAGGER.tag(tokens)

    # HACK this fixes some know failures from postag
    # this is not sustainable but important cases can be added at any time
    # PRs + unittests welcome!
    DETS = ["a", "á", "o", "ós", "aos", "ao"]
    for idx, (w, t) in enumerate(postagged):
        next_w, next_t = postagged[idx + 1] if \
            idx < len(postagged) - 1 else ("", "")

        #  ('á', 'NOUN'), ('Maria', 'NOUN')
        if w.lower() in DETS and t == "NOUN":
            postagged[idx] = (w, "DET")

    return postagged


# word rules for gender
_FEMALE_ENDINGS_PT = ["a", "as"]
_MALE_ENDINGS_PT = ["o", "os"]


def predict_gender_pt(word, text=""):
    # parse gender taking context into account
    word = word.lower()
    words = text.lower().split(" ")
    for idx, w in enumerate(words):
        if w == word and idx != 0:
            # in portuguese usually the previous word (a determinant)
            # assigns gender to the next word
            previous = words[idx - 1].lower()
            if previous in MALE_DETERMINANTS_PT:
                return "male"
            elif previous in FEMALE_DETERMINANTS_PT:
                return "female"

    # get gender using only the individual word
    # see if this word has the gender defined
    if word in GENDERED_WORDS_PT["male"]:
        return "male"
    if word in GENDERED_WORDS_PT["female"]:
        return "female"
    singular = word.rstrip("s")
    if singular in GENDERED_WORDS_PT["male"]:
        return "male"
    if singular in GENDERED_WORDS_PT["female"]:
        return "female"
    # in portuguese the last vowel usually defines the gender of a word
    # the gender of the determinant takes precedence over this rule
    for end_str in _FEMALE_ENDINGS_PT:
        if word.endswith(end_str):
            return "female"
    for end_str in _MALE_ENDINGS_PT:
        if word.endswith(end_str):
            return "male"
    return None
