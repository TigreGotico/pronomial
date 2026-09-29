# Advanced

## How a pronoun gets resolved

Resolution runs in three layers, each exposed as a public call:

1. **`detect_nouns`** POS-tags the sentence and files every noun into role
   buckets: `male`, `female`, `first`, `neutral`, `plural`, `subject`,
   `verb_subject`. Gender comes from `predict_gender`. Plurality and subject
   position come from the tags and word order.
2. **`score_corefs`** walks the sentence, and for each pronoun considers only
   the nouns *before* it. It awards points per matching bucket, then normalizes
   each pronoun's candidate scores to sum to roughly `1.0`.
3. **`link_pronouns`** / **`replace_corefs`** take the single highest-scoring
   candidate per pronoun and either return the link or splice the antecedent
   into the token stream.

The bonuses that shape the ranking:

- **Gender match**: a gendered pronoun lands on a same-gender noun.
- **Freshness**: closer (more recent) antecedents outscore far ones.
- **Verb-subject**: a noun immediately followed by a verb is favored, since
  it is likely the clause subject.
- **Plural**: plural pronouns prefer plural nouns and merged noun groups.

You rarely need to touch the scores, but reading `score_corefs` output is the
fastest way to understand why a given antecedent won.

```python
from pronomial import score_corefs, word_tokenize

text = "Anna met Bob. She waved at him."
tokens = word_tokenize(text)
for pron_idx, cands in score_corefs(text, lang="en").items():
    best = max(cands, key=cands.get)
    print(tokens[pron_idx], "->", tokens[best])
```

## Working with indices safely

`link_pronouns` and `score_corefs` speak token indices, and those indices are
defined by `word_tokenize`, not by Python's `str.split()`. Always tokenize with
the library's tokenizer before indexing into a result:

```python
from pronomial import link_pronouns, word_tokenize

text = "The Romans founded it."
tokens = word_tokenize(text)              # punctuation is its own token
for pron_idx, ante_idx, score in link_pronouns(text, lang="en"):
    print(tokens[pron_idx], "<-", tokens[ante_idx], score)
```

`normalize(text)` gives you the exact spaced form that `replace_corefs` returns,
which is handy when diffing input against output.

## Multi-language notes

Pass any BCP-47 code. Only the segment before `-` is read, so `"pt"` and
`"pt-PT"` behave identically.

```python
from pronomial import replace_corefs

replace_corefs("Juan compró un coche. Él lo conduce.", lang="es")
# 'Juan compró un coche . Él Juan conduce .'
```

- **English** ships the richest rule set: gendered word lists, look-back word
  pairs (`with`/`that`/`in` + follow-ups), and morphological pluralization.
- **Portuguese, Spanish, Catalan** use bundled POS taggers plus
  morphological gender prediction (the noun's article and ending), and a
  trailing-`s` plural test.

The pronoun buckets are the same five everywhere: `male`, `female`, `first`,
`neutral`, `plural`, defined in `pronomial/lang/<code>.py`.

## Gotchas

- **Output is tokenized.** `replace_corefs` returns text with spaces around
  punctuation (`"... millennia ."`). If you need the original spacing, align the
  tokens yourself against `word_tokenize(text)`.
- **A pronoun may resolve to itself.** When no better antecedent scores high
  enough, the top candidate can be the pronoun's own token. The rewrite is then
  a no-op for that slot.
- **Heuristic, not exact.** This is a deliberately small rule-based baseline. It
  shines on short, single-referent sentences and degrades on long sentences with
  competing antecedents or nested clauses. Treat low normalized scores as low
  confidence.
- **Catalan POS tagging is unavailable.** `pos_tag(..., lang="ca")` raises
  `NotImplementedError`, so the full pipeline (`score_corefs`,
  `link_pronouns`, `replace_corefs`) does not run for `ca`. The Catalan pronoun
  and gendered-word lists are still usable on their own.

## Recipe: only rewrite confident links

`replace_corefs` always applies its top pick. To apply a confidence floor,
drive the rewrite yourself from `score_corefs`:

```python
from pronomial import score_corefs, word_tokenize

def replace_above(text, lang="en", threshold=0.5):
    tokens = word_tokenize(text)
    for pron_idx, cands in score_corefs(text, lang=lang).items():
        ante_idx = max(cands, key=cands.get)
        if cands[ante_idx] >= threshold:
            tokens[pron_idx] = tokens[ante_idx]
    return " ".join(tokens)

replace_above("London is old. It was founded by Romans.", threshold=0.5)
```

---
[← API](api.md) · [Home](../readme.md)
