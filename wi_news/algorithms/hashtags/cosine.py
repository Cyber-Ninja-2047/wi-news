#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:58:11 2024

@author: anthony
"""
import numpy as np


def extract_by_cosine(data, n_hashtags=3, **kwargs):
    """
    extract hashtags by cosine similarity

    params
    ------
    data : list,
        the return from `vectorization.preprocess`.
    n_hashtags : int,
        the number of hashtags to extract.

    returns
    -------
    hashtags : list,
        the hashtags of each article.

    """
    return np.array([_extract_one(d, n_hashtags) for d in data], dtype=object)


def _extract_one(data, n_hashtags):
    from wi_news.algorithms.vectorization import NLP

    # get vectors and filter the pool
    pool = np.unique(data.hashtags_pool)
    pool = np.asarray([kw.title() for kw in pool])
    pool, vectors = _get_vectors(pool)

    # get target
    target = NLP(data.text).vector

    # cosine similarities
    cosine = target.dot(vectors.T)
    hashtags = pool[np.argsort(cosine)[-1 : -1 - n_hashtags : -1]]
    return hashtags


def _get_vectors(pool):
    from wi_news.algorithms.vectorization import NLP

    # get vectors
    index_to_remove = []
    vectors = []
    for ind, word in enumerate(pool):
        doc = NLP(word.replace("_", " "))
        if not doc.has_vector:
            index_to_remove.append(ind)
            continue
        vector = doc.vector
        vectors.append(vector / np.linalg.norm(vector))
    vectors = np.asarray(vectors)

    # filter pool
    index = np.ones_like(pool, dtype=bool)
    index[index_to_remove] = False
    pool = pool[index]

    return pool, vectors
