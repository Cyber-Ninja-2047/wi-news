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


def get_news(keywords=None, sort_by="popularity", search_in="title,description",
             page_size=100, language='en',
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
        'sortBy': sort_by,
        'searchIn': search_in,
        'pageSize': page_size,
        'language': language,
    }
    kwargs.update(params)

    response = requests.get(
        URL,
        params=kwargs,
        headers={
            'X-Api-Key': API_KEY,
        },
        timeout=10,
    )
    data = response.json()

    return data
