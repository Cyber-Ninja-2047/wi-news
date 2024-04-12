#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:16:15 2024

@author: anthony
"""
from dataclasses import dataclass
from datetime import datetime
import pytz
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from wi_news.algorithms.vectorization import NLP, LLM, filter_word


def sort_news(query, data, **kwargs):
    "functional sorting"
    sorter = NewsSorter(query, **kwargs)
    data = sorter.sort(data, **kwargs)
    return data


@dataclass
class Query:
    "data of the query"
    text: str
    vector: np.ndarray = None


class NewsSorter:
    """
    sort the news by query in different methods

    params
    ------
    query : str,
       the query keywords
    ngram_range : (min, max),
        minimal and maximal length of ngrams
    date : datetime.datetime or None,
        the date of the search, default is the current date.

    """

    def __init__(self, query, ngram_range=(1, 3), date=None, **kwargs):
        self.ngram_range = ngram_range
        self.query = self._preprocess_query(query)
        if date is None:
            self.date = datetime.now(pytz.utc)

    def get_title(self, data):
        "return the title of each news"
        return [d.raw["title"] for d in data]

    def sort(self, data, method="embeddings", by_date=True, **kwargs):
        """
        sort news data by self.query

        params
        ------
        data : list of Article,
            news data from `vectorization.preprocess`
        method : "embeddings" or "tfidf" or "none",
            sorting methods.
                * embeddings : using sentence-transformer
                * tfidf : using TF-IDF
                * none : do not sort
        by_date : bool,
            sort the news by date or not.

        returns
        -------
        data : list of Article,
            sorted data

        """
        if len(data) == 0:
            return data

        func = getattr(self, f"_score_by_{method}", None)
        if not func:
            raise ValueError(f"invalid value for @method: {method}")
        if self.query.vector is None:
            func = self._score_by_tfidf
        data, scores = func(data, **kwargs)

        if by_date:
            scores /= self._decay_by_date(data)

        return _sort_by_value(data, scores)

    @staticmethod
    def _preprocess_query(query):
        doc = NLP(query)

        text = " ".join((t.lemma_ for t in doc if filter_word(t)))

        vector = None
        if len(doc) > 1:
            vector = LLM.encode([query])[0]

        return Query(text=text, vector=vector)

    def _score_by_none(self, data, **kwargs):
        data = np.asarray(data)
        return data, np.ones(len(data), dtype=float)

    def _score_by_embeddings(self, data, **kwargs):
        "return cosine sim between query and data, higher is better"
        if self.query.vector is None:
            raise ValueError("query doesn't have valid vector")
        data = np.asarray(data)
        vectors = np.asarray([d.vector for d in data])
        cosine = self.query.vector.dot(vectors.T)
        return data, cosine

    def _score_by_tfidf(self, data, **kwargs):
        "return cosine sim between query and data, higher is better"
        ngrams = self.__get_ngrams()

        # compute the TF-IDF matrix
        data = np.asarray(data, dtype=object)
        pipeline = Pipeline(
            [
                ("tf", CountVectorizer(ngram_range=self.ngram_range)),
                # ('scalar', MaxAbsScaler()),
                ("idf", TfidfTransformer()),
            ]
        )
        vectors = pipeline.fit_transform([d.lemma for d in data])
        index = [pipeline['tf'].vocabulary_[k] for k in ngrams]

        weights_query = np.ones_like(index)
        vectors_data = vectors[:, index].toarray()
        cosine = weights_query.dot(vectors_data.T)

        # filter out news has no keywords
        cosine = cosine.flatten()
        mask = cosine > 0
        data = data[mask]
        cosine = cosine[mask]

        # scores for sorting
        return data, cosine

    def _decay_by_date(self, data):
        "return day decays of data, lower is better"
        data = np.asarray(data, dtype=object)

        # plus 1 to decay from 2 days
        days = np.array([(self.date - d.time).days + 1 for d in data])
        days[days < 2] = 2  # for computation stability

        # return for sorting
        decays = np.log2(days)
        return decays


    def __get_ngrams(self):
        ngrams = CountVectorizer(
            ngram_range=self.ngram_range
        ).fit([self.query.text]).vocabulary_
        return list(ngrams)


def _sort_by_value(data, value):
    return data[np.argsort(value)[::-1]]


def _svd(matrix, info_percentage):
    u, sigma, vh = np.linalg.svd(matrix)
    accu_sigma = np.cumsum(sigma)
    index = np.where(accu_sigma > info_percentage)[0][0] + 1
    u = u[:, :index]
    sigma = sigma[:index]
    vh = vh[:, :index]
    return u, sigma, vh
