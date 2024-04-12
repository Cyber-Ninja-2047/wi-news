#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:02:49 2024

@author: anthony
"""
import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from wi_news.algorithms.hashtags import get_pool


REG_OMISSION = re.compile(r"\[\+\d+\schars\]")
REG_REPLACE = re.compile(r"\<.*?\>|\s+")
try:
    NLP = spacy.load("en_core_web_md")
except OSError as exc:
    raise RuntimeError(
        "Please run 'python -m spacy download en_core_web_md' "
        "to download the nlp model."
    ) from exc
LLM = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class Article:
    "data of article"
    raw: dict
    text: str
    lemma: str
    vector: np.ndarray
    hashtags_pool: list
    time: datetime


def filter_word(x):
    "only keep the no-stop words"
    return not x.is_stop and (x.is_alpha or x.is_digit)


def preprocess(article):
    """
    preprocessing for the article

    params
    ------
    article : dict,
        article from news-api,
        including keys: 'title', 'description', and 'content'.

    returns
    -------
    data : Article,
        data of article, including properties below:
        raw : dict,
            raw data from news-api
        text : str,
            combination of title, description and content of the news
        vector : np.array,
            the vector of the article
        hashtags_pool : list,
            possible hashtags

    """
    # init
    text = (
        article["title"]
        + ". "
        + article.get("description", "")
        + ". "
        + article.get("content", "")
    )
    text = REG_REPLACE.sub(" ", text)

    # cut the text
    for mat in REG_OMISSION.finditer(text):
        text = text[: mat.span()[0]]
        break

    doc = NLP(text)

    # lemma
    lemma = ' '.join(map(lambda x: x.lemma_,
                     filter(filter_word, doc)))

    # split into sentences
    sentences = []
    for sent in doc.sents:
        if len(sent) >= 2:
            sentences.append(sent)

    # vector
    vectors = LLM.encode(sentences)
    vectors[0] *= 2
    vector = vectors.sum(0)
    vector /= np.linalg.norm(vector)

    # hashtags pool
    hashtags_pool = get_pool(doc)

    published_at = datetime.fromisoformat(article['publishedAt'])

    return Article(raw=article, text=text, lemma=lemma,
                   vector=vector, hashtags_pool=hashtags_pool,
                   time=published_at)
