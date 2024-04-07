#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:16:12 2024

@author: anthony
"""
import os
import requests


API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    raise ValueError("env: NEWSAPI_KEY is required.")


URL = "https://newsapi.org/v2/everything"


def get_news(keywords=None, sortBy="popularity", searchIn="title,description",
             pageSize=100, language='en',
             **kwargs):
    """
    get news via news api

    kwargs
    ------
    other query parameters of news api.
    please refer https://newsapi.org/docs/endpoints/everything

    """
    params = {
        'q': keywords,
        'sortBy': sortBy,
        'searchIn': searchIn,
        'pageSize': pageSize,
        'language': language,
    }
    kwargs.update(params)

    response = requests.get(
        URL,
        params=kwargs,
        headers={
            'X-Api-Key': API_KEY,
        },
    )
    data = response.json()

    return data
