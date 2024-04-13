#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:57:57 2024

@author: anthony
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def extract_by_tfidf(data, n_hashtags=3, n_words=50, method="tfidf", **kwargs):
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

        * tfidf : returns unique hashtags for each article.
        * tfdf : returns common hashtags for each article.
        * tf : returns hashtags for each article by word frequencies.

    returns
    -------
    hashtags : list,
        the hashtags of each article.

    """
    return _METHOD_TO_FUNC[method](data, n_hashtags, n_words, **kwargs)


def _extract_by_tfidf(data, n_hashtags, n_words, reverse_idf=0, **kwargs):
    pipeline = get_tfidf_pipeline(n_words)

    texts = [" ".join(d.hashtags_pool) for d in data]
    try:
        result = pipeline.fit_transform(texts)
    except ValueError:
        return np.array([[] for _ in data])

    # reverse idf to get origin tf or tfdf
    for _ in range(reverse_idf):
        result /= pipeline["idf"].idf_

    words = sorted(
        [(y, x.title()) for x, y in pipeline["tf"].vocabulary_.items()]
    )
    words = np.array(words).T[1]

    # get hashtags
    hashtags = words[result.toarray().argsort()[:, -1 : -1 - n_hashtags : -1]]
    return hashtags


def get_tfidf_pipeline(n_words):
    "return the tfidf pipeline"
    pipeline = Pipeline(
        [
            (
                "tf",
                CountVectorizer(
                    max_features=n_words, token_pattern=r"(?u)\b\w[\w\-]+\b"
                ),
            ),
            ("idf", TfidfTransformer()),
        ]
    )
    return pipeline


def _extract_by_tf(data, n_hashtags, n_words, **kwargs):
    return _extract_by_tfidf(data, n_hashtags, n_words, 1, **kwargs)


def _extract_by_tfdf(data, n_hashtags, n_words, **kwargs):
    return _extract_by_tfidf(data, n_hashtags, n_words, 2, **kwargs)


_METHOD_TO_FUNC = {
    "tfidf": _extract_by_tfidf,
    "tfdf": _extract_by_tfdf,
    "tf": _extract_by_tf,
}
