#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:53:30 2024

@author: anthony
"""
import os
import numpy as np
from flask import Flask, request, jsonify
from wi_news.algorithms.get_news import get_news
from wi_news.algorithms.vectorization import preprocess
# from wi_news.algorithms.classification import classify
from wi_news.algorithms.clustering import cluster_data
from wi_news.algorithms.sort_news import sort_news
from wi_news.algorithms.hashtags import extract_by_tfidf, extract_for_clusters


STATIC_PATH = os.path.join(os.path.split(__file__)[0], "static")

app = Flask(
    "wi-news-server",
    static_url_path="/",
    static_folder=STATIC_PATH,
)


@app.route("/api/get_news")
def api_get_news():
    "the news preprocessing pipeline"
    query = request.args.get("q")
    if not query:
        return jsonify({"message": "Please input some keywords."}), 400

    # vectorization
    articles = [
        a for a in get_news(query)["articles"]
        if a["title"] and a["title"] != "[Removed]"
    ]
    if len(articles) == 0:
        return jsonify({"message": "No news found."}), 404
    data = np.asarray([preprocess(a) for a in articles], object)

    # sort
    data = sort_news(query, data)

    # classification
    # datas = classify(data)
    datas = {"All News": data}  # mute the classifier because of low performance

    # clustering
    clusters = {}
    dict_data = {}
    to_return = {"clusters": clusters, "data": dict_data}
    for category, data in datas.items():
        if len(data) == 0:
            continue
        model = cluster_data(data)
        labels = model.labels_

        # hashtags for the cluster
        hashtags = extract_for_clusters(data, labels)
        cluster_meta = {
            "labels": np.unique(labels).tolist(),
            "hashtags": hashtags.tolist(),
        }
        clusters[category] = cluster_meta

        # hashtags for articles
        hashtaged_articles = []
        dict_data[category] = hashtaged_articles
        for label in np.unique(labels):
            sub_data = data[label == labels]
            sub_hashtags = extract_by_tfidf(sub_data)
            label = int(label)
            for sdata, shash in zip(sub_data, sub_hashtags):
                sdata.raw.update(
                    {
                        "category": category,
                        "cluster": label,
                        "hashtags": shash.tolist(),
                    }
                )
                hashtaged_articles.append(sdata.raw)

    return jsonify(to_return), 200
