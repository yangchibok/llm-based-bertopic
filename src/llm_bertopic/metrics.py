"""NPMI/coherence, extrapolated IRBO and OCTIS-compatible smoothed KL uniform.

The formulas are implemented here to avoid the unrelated training dependencies of OCTIS.
See read.md for metric definitions and execution notes.
"""

from itertools import combinations

import numpy as np
from scipy import sparse


def inverted_rbo(topics, topk, weight=0.9):
    if len(topics) < 2:
        return float("nan")
    if not 0 < weight < 1:
        raise ValueError("RBO weight must lie between 0 and 1")
    lists = [t[:topk] for t in topics]
    if any(len(t) < topk or len(set(t)) != topk for t in lists):
        return float("nan")
    scores = []
    for left, right in combinations(lists, 2):
        a, b, overlap, weighted = set(), set(), 0.0, 0.0
        for depth, (x, y) in enumerate(zip(left, right), 1):
            a.add(x)
            b.add(y)
            overlap = len(a & b) / depth
            weighted += (1 - weight) * weight ** (depth - 1) * overlap
        scores.append(weighted + weight ** topk * overlap)
    return float(np.clip(1 - np.mean(scores), 0, 1))


def kl_uniform(matrix, epsilon=1e-5):
    """Mean D(P+epsilon || Uniform+epsilon); match OCTIS's non-renormalized smoothing."""
    if len(matrix.shape) != 2 or min(matrix.shape) == 0:
        return float("nan")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    rows, columns = matrix.shape
    scores = []
    # One dense row at a time: avoid densifying the entire topic x vocabulary matrix.
    for i in range(rows):
        row = matrix.getrow(i).toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix[i]).ravel()
        row = row.astype(float)
        if not np.isfinite(row).all() or (row < 0).any():
            raise ValueError("KL requires finite nonnegative topic weights")
        if not row.any():
            scores.append(0.0)
            continue
        p = row / row.sum() + epsilon
        q = 1.0 / columns + epsilon
        positive = p > 0
        scores.append(float(np.sum(p[positive] * np.log(p[positive] / q))))
    return float(np.mean(scores))


def evaluate_coherence(topics, texts, settings, lda_model=None, dictionary=None):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    tokens = [t.split() for t in texts]
    dictionary = dictionary if dictionary is not None else Dictionary(tokens)
    kwargs = {"texts": tokens, "dictionary": dictionary,
              "corpus": [dictionary.doc2bow(t) for t in tokens],
              "coherence": settings["coherence"], "processes": settings["processes"]}
    if settings.get("window_size") is not None:
        kwargs["window_size"] = settings["window_size"]
    if lda_model is not None:
        kwargs.update(model=lda_model, topn=settings["lda_coherence_topn"])
        return float(CoherenceModel(**kwargs).get_coherence()), 0
    # Historical code built a whitespace dictionary, so phrases can be out of vocabulary.
    # Explicitly expose the ignored word count and don't fabricate coherence for empty topics.
    known = [[w for w in topic if w in dictionary.token2id] for topic in topics]
    ignored = sum(len(t) - len(k) for t, k in zip(topics, known))
    if not known or any(len(words) < 2 for words in known):
        return float("nan"), ignored
    return float(CoherenceModel(topics=known, **kwargs).get_coherence()), ignored


def evaluate_notebook(model, texts, topk, settings):
    """Cells 16/18 of bertopic_compare_results.ipynb, using the actual OCTIS metrics.

    Preserve topic order, empty word entries, outlier handling and dense matrix dtype.
    No reference scores are read here: all three values are calculated from this fit.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from octis.evaluation_metrics.diversity_metrics import InvertedRBO
    from octis.evaluation_metrics.topic_significance_metrics import KL_uniform
    topic_map = model.get_topics()
    topic_words = [[word for word, _ in topic_map[k]] for k in topic_map if k != -1]
    tokens = [text.split() for text in texts]
    dictionary = Dictionary(tokens)
    options = {"coherence": settings["coherence"]}
    # -1 leaves Gensim's original default (cpu_count - 1) unchanged.
    if settings["processes"] != -1:
        options["processes"] = settings["processes"]
    if settings.get("window_size") is not None:
        options["window_size"] = settings["window_size"]
    coherence = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary,
                               corpus=[dictionary.doc2bow(text) for text in tokens], **options).get_coherence()
    rbo_words = [[word for word, _ in topic] for topic in topic_map.values() if len(topic) > 0]
    return {
        "coherence_score": float(coherence),
        "inverted_rbo_score": float(InvertedRBO(topk=topk, weight=settings["rbo_weight"]).score(
            {"topics": rbo_words})),
        "kl_uniform_score": float(KL_uniform().score({"topic-word-matrix": np.array(model.c_tf_idf_.todense())})),
        "coherence_ignored_words": sum(word not in dictionary.token2id for words in topic_words for word in words),
    }
