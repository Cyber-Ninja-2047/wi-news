#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:03:43 2024

@author: kibtia
"""
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA


_NAME_TO_CLAZZ = {
    "kmeans": KMeans,
    "gmm": GaussianMixture,
    "agglomerative": AgglomerativeClustering,
}

@dataclass
class FakeModel:
    labels_ : list


def cluster_data(data, methods=("kmeans",), k_range=(2, 11)):
    """
    cluster with different different parameters.
    return best_model.labels_

    """
    length = len(data)
    if len(data) < 2:
        return FakeModel(labels_=np.zeros_like(data))

    # scale the data
    samples = np.asarray([d.vector for d in data])
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)

    # clustering on the data

    if isinstance(methods, str):
        methods = [methods]

    # find best for each method
    model_scores = []
    for method in methods:
        models = [
            _get_model(method, n_clusters=k).fit(samples)
            for k in range(*k_range) if k <= length
        ]

        # evaluations
        scores = [_evaluate(m, samples) for m in models]
        pos = np.argmax(scores)
        # pos = _find_local_maximum(scores)

        model_scores.append((models[pos], scores[pos]))

    # find the best at all
    best_model, _ = max(model_scores, key=lambda x: x[1])

    return best_model


def reduce_dimensionality(samples, n_dim=2):
    "reduce the samples' dimensionality"
    return PCA(n_components=n_dim).fit_transform(samples)


def _get_model(method, n_clusters, **kwargs):
    if method == "gmm":
        kwargs["n_components"] = n_clusters
    else:
        kwargs["n_clusters"] = n_clusters
    return _NAME_TO_CLAZZ[method](**kwargs)


def _evaluate(model, samples):
    return silhouette_score(samples, model.labels_)


def _find_local_maximum(scores):
    length = len(scores)
    pos = 0
    for i in range(length):
        if i - 1 < 0 or i + 1 >= length:
            continue
        if scores[i - 1] < scores[i] > scores[i + 1]:
            pos = i
            break
    return pos
