#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:12:02 2024

@author: anthony
"""
import os
import pickle
import numpy as np


path = os.path.split(__file__)[0]


with open(os.path.join(path, "prep.pkl"), "rb") as f:
    PREP = pickle.load(f)

with open(os.path.join(path, "clf.pkl"), "rb") as f:
    CLASSIFIER = pickle.load(f)


CATEGORIES = ["Others", "POLITICS", "SPORTS", "COMEDY"]
CATEGORIES = [c.title() for c in CATEGORIES]


def classify(articles):
    """
    classify the articles

    returns
    -------
    result : dict
        {category : [article, ...]}

    """
    # classification
    articles = np.asarray(articles)
    data = np.asarray([d.vector for d in articles])
    data = PREP.transform(data)
    labels = CLASSIFIER.predict(data)

    # result
    result = {}
    for label in np.unique(labels):
        category = CATEGORIES[int(label)]
        result[category] = articles[label == labels]
    return result
