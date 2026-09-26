from pronomial.utils import predict_gender, pos_tag, word_tokenize, is_plural


class PronomialCoreferenceSolver:
    @staticmethod
    def _load_lang_resources(lang):
        # universal tagset, depends on pos tagger model for each lang
        PRONOUN_TAG = ['PRON']
        NOUN_TAG = ['NOUN']
        JJ_TAG = ['ADJ']
        PLURAL_NOUN_TAG = ['NOUN']
        SUBJ_TAG = ['NOUN']

        # this is a basic mechanism to allow to "look back", the naming
        # reflects what this does poorly, each language can define word
        # pairs that will give preference to an earlier noun in the
        # sentence, loosely named after a couple examples in the
        # english unittests. TODO generalize this
        WITH = WITH_FOLLOWUP = THAT = THAT_FOLLOWUP = IN = IN_FOLLOWUP = []

        NEUTRAL_WORDS = []

        # symbol used when merging Nouns to replace plural pronouns
        NAME_JOINER = "+"

        # word lists
        PRONOUNS = {}
        GENDERED_WORDS = {}

        # Load resources for selected lang
        if lang.startswith("en"):
            from pronomial.lang.en import NOUN_TAG_EN, PLURAL_NOUN_TAG_EN, \
                PRONOUNS_EN, PRONOUN_TAG_EN, SUBJ_TAG_EN, JJ_TAG_EN, WITH_EN, \
                GENDERED_WORDS_EN, WITH_FOLLOWUP_EN, THAT_EN, \
                THAT_FOLLOWUP_EN, NEUTRAL_WORDS_EN, NAME_JOINER_EN, IN_EN, \
                IN_FOLLOWUP_EN
            GENDERED_WORDS = GENDERED_WORDS_EN
            NOUN_TAG = NOUN_TAG_EN
            SUBJ_TAG = SUBJ_TAG_EN
            PRONOUN_TAG = PRONOUN_TAG_EN
            PRONOUNS = PRONOUNS_EN
            PLURAL_NOUN_TAG = PLURAL_NOUN_TAG_EN
            JJ_TAG = JJ_TAG_EN
            WITH = WITH_EN
            WITH_FOLLOWUP = WITH_FOLLOWUP_EN
            THAT = THAT_EN
            THAT_FOLLOWUP = THAT_FOLLOWUP_EN
            NEUTRAL_WORDS = NEUTRAL_WORDS_EN
            NAME_JOINER = NAME_JOINER_EN
            IN = IN_EN
            IN_FOLLOWUP = IN_FOLLOWUP_EN
        elif lang.startswith("pt"):
            from pronomial.lang.pt import PRONOUNS_PT, GENDERED_WORDS_PT
            GENDERED_WORDS = GENDERED_WORDS_PT
            PRONOUNS = PRONOUNS_PT
        elif lang.startswith("es"):
            from pronomial.lang.es import PRONOUNS_ES, GENDERED_WORDS_ES
            GENDERED_WORDS = GENDERED_WORDS_ES
            PRONOUNS = PRONOUNS_ES
        elif lang.startswith("ca"):
            from pronomial.lang.ca import PRONOUNS_CA, GENDERED_WORDS_CA
            GENDERED_WORDS = GENDERED_WORDS_CA
            PRONOUNS = PRONOUNS_CA

        return PRONOUN_TAG, NOUN_TAG, JJ_TAG, PLURAL_NOUN_TAG, SUBJ_TAG, \
               WITH, WITH_FOLLOWUP, THAT, THAT_FOLLOWUP, IN, IN_FOLLOWUP, \
               NEUTRAL_WORDS, NAME_JOINER, PRONOUNS, \
               GENDERED_WORDS

    @staticmethod
    def detect_nouns(sentence, lang="en", return_idx=True):

        PRONOUN_TAG, NOUN_TAG, JJ_TAG, PLURAL_NOUN_TAG, SUBJ_TAG, \
        WITH, WITH_FOLLOWUP, THAT, THAT_FOLLOWUP, IN, IN_FOLLOWUP, \
        NEUTRAL_WORDS, NAME_JOINER, PRONOUNS, GENDERED_WORDS = \
            PronomialCoreferenceSolver._load_lang_resources(lang)

        tags = pos_tag(sentence, lang=lang)
        prev_names_idx = {
            "male": [],
            "female": [],
            "first": [],
            "neutral": [],
            "plural": [],
            "subject": [],
            "verb_subject": []
        }

        for idx, (w, t) in enumerate(tags):
            next_w, next_t = tags[idx + 1] if idx < len(tags) - 1 else ("", "")
            prev_w, prev_t = tags[idx - 1] if idx > 0 else ("", "")
            if t in NOUN_TAG:
                if prev_w in NEUTRAL_WORDS:
                    prev_names_idx["neutral"].append(idx)
                else:
                    gender = predict_gender(w, prev_w, lang=lang)
                    if w in PRONOUNS["female"] or \
                            w.lower() in GENDERED_WORDS["female"]:
                        prev_names_idx["female"].append(idx)
                    elif w in PRONOUNS["male"] or \
                            w.lower() in GENDERED_WORDS["male"]:
                        prev_names_idx["male"].append(idx)
                    elif w[0].isupper() or prev_t in ["DET"]:
                        prev_names_idx[gender].append(idx)
                    prev_names_idx["subject"].append(idx)
                    prev_names_idx["neutral"].append(idx)

                if next_t.startswith("V") and not prev_t.startswith("V"):
                    prev_names_idx["verb_subject"].append(idx)

            elif t in SUBJ_TAG:
                prev_names_idx["subject"].append(idx)
                gender = predict_gender(w, prev_w, lang=lang)
                prev_names_idx["neutral"].append(idx)
                if gender == "female":
                    prev_names_idx["female"].append(idx)
                if gender == "male":
                    prev_names_idx["male"].append(idx)
            elif t in PLURAL_NOUN_TAG:
                prev_names_idx["plural"].append(idx)
                if w[0].isupper():
                    gender = predict_gender(w, prev_w, lang=lang)
                    if not prev_names_idx[gender]:
                        prev_names_idx[gender].append(idx)
            elif t in JJ_TAG:  # common tagger error
                if w[0].isupper():
                    gender = predict_gender(w, prev_w, lang=lang)
                    if not prev_names_idx[gender]:
                        prev_names_idx[gender].append(idx)
                if not prev_names_idx["neutral"]:
                    prev_names_idx["neutral"].append(idx)
                if not prev_names_idx["subject"]:
                    prev_names_idx["subject"].append(idx)

        if isinstance(sentence, str):
            tokens = word_tokenize(sentence, lang=lang)
        else:
            tokens = sentence

        if not return_idx:
            prev_names_idx = {k: [tokens[i] for i in v]
                              for k, v in prev_names_idx.items()}
        prev_names_idx["tokens"] = tokens
        return prev_names_idx

    @staticmethod
    def score_corefs(sentence, lang="en"):
        PRONOUN_TAG, NOUN_TAG, JJ_TAG, PLURAL_NOUN_TAG, SUBJ_TAG, \
        WITH, WITH_FOLLOWUP, THAT, THAT_FOLLOWUP, IN, IN_FOLLOWUP, \
        NEUTRAL_WORDS, NAME_JOINER, PRONOUNS, \
        GENDERED_WORDS = PronomialCoreferenceSolver._load_lang_resources(lang)

        tags = pos_tag(sentence, lang=lang)
        pron_list = [p for k, p in PRONOUNS.items()]
        flatten = lambda l: [item for sublist in l for item in sublist]
        pron_list = flatten(pron_list)

        tagged_nouns = PronomialCoreferenceSolver.detect_nouns(sentence, lang)
        tokens = tagged_nouns.pop("tokens")
        candidates = {}

        for idx, (w, t) in enumerate(tags):
            if (t in PRONOUN_TAG and w.lower() in pron_list) or \
                    any(w in items for k, items in PRONOUNS.items()):
                candidates[idx] = {}

                prev_w, prev_t = tags[idx - 1] if idx > 0 else ("", "")
                prev_ids = {k: [i for i in v if i <= idx]
                            for k, v in tagged_nouns.items()}
                prev_names = {k: [tokens[i] for i in v]
                              for k, v in prev_ids.items()}
                # this is a basic mechanism to allow to "look back", the naming
                # reflects what this does poorly, each language can define word
                # pairs that will give preference to an earlier noun in the
                # sentence, loosely named after a couple examples in the
                # english unittests. TODO generalize this
                idz = -1
                if prev_w.lower() in WITH and w.lower() in WITH_FOLLOWUP:
                    idz = -2
                elif prev_w.lower() in THAT and w.lower() in THAT_FOLLOWUP:
                    idz = -2
                elif prev_w.lower() in IN and w.lower() in IN_FOLLOWUP:
                    idz = -2
                wl = w.lower()

                # score subject tags
                if wl in PRONOUNS["neutral"] or \
                        wl in PRONOUNS["male"] or \
                        wl in PRONOUNS["female"] or \
                        (wl in PRONOUNS["plural"] and t in ["WP"]):  # "who"

                    if prev_names["subject"]:
                        if idz < 0:
                            idz = len(prev_names["subject"]) + idz
                        for ids, x in enumerate(prev_ids["subject"]):
                            if x not in candidates[idx]:
                                candidates[idx][x] = 0
                            if ids == idz:
                                candidates[idx][x] += 10  # freshness bonus
                            candidates[idx][x] += ids

                # score gendered pronoun tags
                if wl in PRONOUNS["male"]:
                    if prev_names["male"]:
                        if idz < 0:
                            idz = len(prev_names["male"]) + idz
                        for ids, x in enumerate(prev_ids["male"]):
                            if x not in candidates[idx]:
                                candidates[idx][x] = 0
                            candidates[idx][x] += 20  # gender match
                            # freshness bonus
                            candidates[idx][x] += ids * 10
                            if ids == idz:
                                candidates[idx][x] += 10
                            if tokens[x] in [_ for _ in prev_names["male"]
                                             if _ in prev_names["verb_subject"]]:
                                candidates[idx][x] += 25  # verb subject match

                elif wl in PRONOUNS["female"]:
                    if prev_names["female"]:
                        if idz < 0:
                            idz = len(prev_names["female"]) + idz
                        for ids, x in enumerate(prev_ids["female"]):
                            if x not in candidates[idx]:
                                candidates[idx][x] = 0

                            candidates[idx][x] += 20  # gender match
                            # freshness bonus
                            candidates[idx][x] += ids * 10
                            if ids == idz:
                                candidates[idx][x] += 10
                            if tokens[x] in [_ for _ in prev_names["female"]
                                             if _ in prev_names["verb_subject"]]:
                                candidates[idx][x] += 25  # verb subject match

                elif wl in PRONOUNS["neutral"]:
                    if prev_names["neutral"]:
                        for ids, x in enumerate(prev_ids["neutral"]):
                            if x not in candidates[idx]:
                                candidates[idx][x] = 0
                            # freshness bonus
                            candidates[idx][x] += ids

                            if tokens[x] in prev_names["verb_subject"]:
                                candidates[idx][x] += 20  # verb subject match

                # score plural tags
                elif wl in PRONOUNS["plural"]:
                    if prev_names["plural"]:
                        if idz < 0:
                            idz = len(prev_names["plural"]) + idz
                        for ids, x in enumerate(prev_ids["plural"]):
                            if x not in candidates[idx]:
                                candidates[idx][x] = 0
                            candidates[idx][x] += 20  # plural pronoun match
                            # freshness bonus
                            candidates[idx][x] += ids * 10
                            if ids == idz:
                                candidates[idx][x] += 10
                            if tokens[x] in [_ for _ in prev_names["subject"] if
                                             is_plural(_, lang)]:
                                candidates[idx][x] += 15  # plural word match
                            if ids in prev_ids["verb_subject"]:
                                candidates[idx][x] += 10  # verb subject match

                # catch all scorer
                # nothing matched but we have a pronoun, if it's not
                # a first person pronoun let's score all subject tags
                elif prev_names["subject"] and wl not in PRONOUNS["first"]:
                    if idz < 0:
                        idz = len(prev_names["subject"]) + idz
                    for ids, x in enumerate(prev_ids["subject"]):
                        if x not in candidates[idx]:
                            candidates[idx][x] = 0
                        if ids == idz:
                            candidates[idx][x] += 10  # freshness bonus
                        candidates[idx][x] += ids

        # make score of all results add up to 1
        for tok_idx in candidates:
            total = sum(
                score for tok2_idx, score in candidates[tok_idx].items()) + \
                    0.001  # ensure no division by 0
            for tok2_idx, score in candidates[tok_idx].items():
                candidates[tok_idx][tok2_idx] = round(score / total, 2)
        return {k: {k2: v2 for k2, v2 in v.items() if v2 > 0}
                for k, v in candidates.items()}

    @staticmethod
    def solve_corefs(sentence, lang="en", return_idx=True):
        tokens = word_tokenize(sentence, lang=lang)
        candidates = PronomialCoreferenceSolver.score_corefs(sentence, lang)
        corefs = []
        for tok_id, match in candidates.items():
            matches = [(tok_id, mtok_id, score)
                       for mtok_id, score in match.items()]
            matches = sorted(matches, key=lambda k: k[2], reverse=True)
            if len(matches):
                corefs.append(matches[0])
        if return_idx:
            return corefs
        return [(tokens[tok_id], tokens[mtok_id], score)
                for tok_id, mtok_id, score in corefs]

    @classmethod
    def replace_corefs(cls, text, lang="en"):
        tokens = word_tokenize(text, lang=lang)
        for tok_id, mtok_id, score in cls.solve_corefs(text, lang=lang):
            tokens[tok_id] = tokens[mtok_id]
        return " ".join(tokens)

    @staticmethod
    def normalize(text, lang="en"):
        return " ".join(word_tokenize(text, lang=lang))


def normalize(*args, **kwargs):
    return PronomialCoreferenceSolver.normalize(*args, **kwargs)


def detect_nouns(*args, **kwargs):
    return PronomialCoreferenceSolver.detect_nouns(*args, **kwargs)


def link_pronouns(*args, **kwargs):
    return PronomialCoreferenceSolver.solve_corefs(*args, **kwargs)


def score_corefs(*args, **kwargs):
    return PronomialCoreferenceSolver.score_corefs(*args, **kwargs)


def replace_corefs(*args, **kwargs):
    return PronomialCoreferenceSolver.replace_corefs(*args, **kwargs)

