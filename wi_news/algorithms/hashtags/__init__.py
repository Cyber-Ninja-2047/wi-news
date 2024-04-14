#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:17:56 2024

@author: anthony
"""
from queue import Queue
from treelib import Tree
import numpy as np
from dataclasses import dataclass
from pyinflect import INFLECTION_INST
from .cosine import extract_by_cosine
from .tf_idf import extract_by_tfidf, get_tfidf_pipeline


_NAME_TO_METHOD = {
    "tfidf": extract_by_tfidf,
    "cosine": extract_by_cosine,
}


@dataclass
class _BaseHashtags:
    hashtags_pool: list


def extract_for_clusters(data, labels, n_hashtags=3, n_words=50, **kwargs):
    data = np.asarray(data, object)

    # get most common words for each clusters
    clusters = []
    for lab in np.unique(labels):
        subset = data[lab == labels]
        hashtags = extract_by_tfidf(
            subset,
            method="tfdf",
            n_hashtags=n_hashtags,
            n_words=n_words,
            **kwargs,
        ).flatten()
        clusters.append(_BaseHashtags(hashtags_pool=hashtags))

    hashtags = extract_by_tfidf(
        clusters, n_hashtags=n_hashtags, n_words=n_words
    )
    return hashtags


def get_pool(doc):
    """
    get hashtags pool from a document.

    returns
    -------
    pool : list of str

    """
    pool = []

    for sent in doc.sents:
        extractor = KeywordsExtractor()
        pool.extend(extractor.find(sent))

    return pool


class KeywordsExtractor:
    "Extract most promising keywords from a sentence"

    __target_dep = frozenset(["nsubj", "pobj", "dobj"])

    __subtree_dep = frozenset(["compound", "amod", "nummod"])

    def __init__(self):
        self.__targets = set()
        self.queue = Queue()
        self.tree_ = Tree()
        self.pool_ = []
        self.__fitted = False

    def find(self, sent):
        """
        extract promising keywords from the sentance.

        returns
        -------
        pool : list,
            the pool of the keywords

        """
        # check state
        if self.__fitted:
            raise RuntimeError(
                f"Every {type(self).__name__} can only be fitted once"
            )
        self.__fitted = True

        # build tree
        self._build_tree(sent)

        # put keywords into pool
        for target in self.__targets:
            keyword = self.__get_keyword(target)
            self.pool_.append(keyword)

        # get keywords from verb root
        self.pool_.extend(self._extract_from_verb(sent.root))

        return self.pool_

    def _build_tree(self, sent):
        self.queue.put((None, sent.root))
        while not self.queue.empty():
            self.__read_next_token()

    def __get_conjs(self, token):
        to_return = [token]
        for child in token.children:
            if child.dep_ == "conj":
                to_return.append(child)
        return to_return

    def _extract_from_verb(self, token):
        # a special rule for verb
        if not token.pos_ == "VERB":
            return []

        event = INFLECTION_INST.spacyGetInfl(token, "VBG")
        keywords = []
        objs = filter(
            lambda x: "dobj" in x.dep_ and not x.is_stop, token.children
        )
        for obj in objs:
            for conj in self.__get_conjs(obj):
                keywords.append(f"{conj.lemma_}_{event}")
        return keywords

    def __justify_target(self, parent, token):
        is_target = (
            any((x in token.dep_ for x in self.__target_dep))
            or (token.dep_ == "conj" and parent in self.__targets)
            or (token.dep_ == "attr" and token.pos_ == "NOUN")
        )
        return is_target and not token.is_stop

    def __read_next_token(self):
        parent, token = self.queue.get()

        if self.__justify_target(parent, token):
            self.__targets.add(token)
        self.tree_.create_node(
            f'{token.text} {token.pos_} {token.dep_} {"stopword" if token.is_stop else ""}',
            token,
            parent=parent,
        )
        for child in token.children:
            self.queue.put((token, child))

    def __get_subtree(self, target, root=False):
        # get the subtree of target
        to_return = []
        if root or target.dep_ in self.__subtree_dep:
            to_return.append(target)
        else:
            return to_return

        for child in target.children:
            to_return.extend(self.__get_subtree(child))
        return to_return

    def __get_keyword(self, target):
        subtree = self.__get_subtree(target, True)

        # sort by position and remove stopwords
        subtree = sorted(
            filter(lambda x: not x.is_stop, subtree), key=lambda x: x.i
        )
        index_target = subtree.index(target)

        # remove modifiers before compound
        for ind in range(index_target):
            if subtree[ind].dep_ == "compound":
                subtree = subtree[ind:]
                index_target -= ind
                break

        return "_".join((x.lemma_ for x in subtree))


def _filter_by_ents(ind, token, ents):
    for ent in ents:
        if ent.start <= ind < ent.end:
            return False
    return True


def _filter_by_stop_punct(token):
    return not (token.is_stop or token.is_punct)


def _filter_by_dep(token):
    return "subj" in token.dep_ or "obj" in token.dep_


def _filter_by_pos(token):
    return token.pos_ in {"PROPN", "NOUN"}


def _convert_by_dep(ind, token, doc):
    prev = doc[ind - 1]
    if prev.dep_ == "amod":
        return f"{prev.text}_{token.text}"
    return token.lemma_
