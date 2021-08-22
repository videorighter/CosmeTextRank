# -*- coding: utf-8 -*-
"""
Author : videorighter
"""

from textrank.rank2vec import pagerank
from textrank.word import word_graph


class KeywordSummarizer:

    def __init__(self, sents=None, tokenizer=None, min_count=2, window=-1, min_cooccurrence=2,
                 vocab2idx=None, bl_ratio=0.85, max_iter=30, verbose=True):
        self.tokenizer = tokenizer
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab2idx = vocab2idx
        self.bl_ratio = bl_ratio
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        if you want to use custom bias, use this function
        :param sents: list of str
            sentence list
        :param bias: None or numpy.ndarray
            page rank bias term
        :return: None
        """
        g, self.idx2vocab = word_graph(sents, self.tokenizer, self.min_count, self.window, self.min_cooccurrence,
                                       verbose=self.vocab2idx)
        self.R = pagerank(g, self.bl_ratio, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print(f'trained TextRank. n words = {self.R.shape[0]}')

    def keywords(self, top_keywords=30):
        """
        returns keywords and rank scores in descending order by the number of top_keywords
        :param top_keywords: int
            number of keywords selected from TextRank
        :return: list of tuple
            each tuple stands for (word, rank_score)
        """
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')
        idxs = self.R.argsort()[-top_keywords:]  # 오름차순으로 마지막 top_keywords 수 만큼
        keywords = [(self.idx2vocab[idx], self.R[idx]) for idx in reversed(idxs)]  # 이걸 다시 내림차순으로
        return keywords

    def summarize(self, sents, top_keywords=30):
        """
        returns keywords as summarized keywords
        :param sents: list of str
            sentence list
        :param top_keywords: int
            number of keywords selected from TextRank
        :return: list of tuple
            each tuple stands for (word, wank)
        """
        self.train_textrank(sents)
        return self.keywords(top_keywords)

