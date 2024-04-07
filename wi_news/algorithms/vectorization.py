#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:02:49 2024

@author: anthony
"""
import re
from queue import deque
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from wi_news.algorithms.hashtags_extraction import get_pool


REG_OMISSION = re.compile(r'\[\+\d+\schars\]')
REG_HTMLTAG = re.compile(r'\<.*?\>')
try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError("Please run 'python -m spacy download en_core_web_md' "
                       "to download the nlp model.")
LLM = SentenceTransformer('all-MiniLM-L6-v2')


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
    vector : np.array,
        the vector of the article
    hashtags_pool : list,
        possible hashtags

    """
    # init
    text = (article['title'] + '. '+
            article.get('description', '') + '. ' +
            article.get('content', ''))
    text = REG_HTMLTAG.sub(' ', text)

    # cut the text
    for mat in REG_OMISSION.finditer(text):
        text = text[: mat.span()[0]]
        break

    doc = NLP(text)

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

    return vector, hashtags_pool
