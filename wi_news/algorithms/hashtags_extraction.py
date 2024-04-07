#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:57:57 2024

@author: anthony
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def extract_hashtags(data, n_hashtags=3, n_words=50, method="tfidf"):
    """
    extract hashtags from processed data.

    params
    ------
    data : list,
        the return from `vectorization.preprocess`.
    n_hashtags : int,
        the number of hashtags to extract.
    n_words : int,
        the size of words pool.
    method : str,
        extraction method, can be one of:
            "tfidf", "tfdf"

        * tfidf returns unique hashtags for each article.
        * tfdf returns common hashtags for each article.

    returns
    -------
    hashtags : list,
        the hashtags of each article.

    """
    return _METHOD_TO_FUNC[method](data, n_hashtags, n_words)


def get_pool(doc):
    """
    get hashtags pool from a document.

    returns
    -------
    pool : list of str

    """
    pool = []

    # entries
    for ent in doc.ents:
        pool.append(ent.text.replace(" ", "_"))

    # subj and obj
    for ind, token in enumerate(doc):
        if (_filter_by_ents(ind, token, doc.ents) and
            _filter_by_stop_punct(token) and
            (_filter_by_dep(token) or
             _filter_by_pos(token))):
            pool.append(_convert_by_dep(ind, token, doc))

    return pool


def _filter_by_ents(ind, token, ents):
    for ent in ents:
        if ent.start <= ind < ent.end:
            return False
    return True

def _filter_by_stop_punct(token):
    return not (token.is_stop or token.is_punct)

def _filter_by_dep(token):
    return "subj" in token.dep_ or "obj" in token.dep_


def _filter_by_pos(token):
    return token.pos_ in {'PROPN', "NOUN"}


def _convert_by_dep(ind, token, doc):
    prev = doc[ind - 1]
    if prev.dep_ == "amod":
        return f"{prev.text}_{token.text}"
    return token.lemma_


def _extract_by_tfidf(data, n_hashtags, n_words, tfdf=False):
    pipeline = Pipeline(
        [
            (
                "tf",
                CountVectorizer(
                    max_features=50, token_pattern=r"(?u)\b\w[\w\-]+\b"
                ),
            ),
            ("idf", TfidfTransformer()),
        ]
    )

    texts = [" ".join(d[1]) for d in data]
    result = pipeline.fit_transform(texts)

    if tfdf:
        result /= pipeline["idf"].idf_ ** 2

    words = sorted([(y, x) for x, y in pipeline["tf"].vocabulary_.items()])
    words = np.array(words).T[1]

    # get hashtags
    hashtags = words[result.toarray().argsort()[:, -1 : -1 - n_hashtags : -1]]
    return hashtags


def _extract_by_tfdf(data, n_hashtags, n_words):
    return _extract_by_tfidf(data, n_hashtags, n_words, True)


_METHOD_TO_FUNC = {
    "tfidf": _extract_by_tfidf,
    "tfdf": _extract_by_tfdf,
}
