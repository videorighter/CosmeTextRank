# -*- coding: utf-8 -*-
"""
Author : videorighter
"""
from collections import Counter
from scipy.sparse import csr_matrix


def vocab_indexing(sents, tokenizer, min_count):
    """
    :param sents: list of str
        sentence list
    :param tokenizer: callable
        tokenizer that returns list of words including Noun, Verb, Adjective, Stem
    :param min_count: int
        minimum number of frequency
    :return idx2vocab: list of str
        most used word order
    :return vocab2idx: dict
        word-index dictionary
    """
    # 문장마다 토크나이징 -> 각 형태소의 등장 수
    counter = Counter(word for sent in sents for word in tokenizer(sent))
    # 최소 개수 이상만
    counter = {word: count for word, count in counter.items() if count >= min_count}
    idx2vocab = [word for word, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(idx2vocab)}
    return idx2vocab, vocab2idx


def dict2mat(dict, n_rows, n_cols):
    """
    dictionary to vector
    :param dict: dict
        {(idx 1, idx 2): sparse number}
    :param n_rows: int
        cooccurrence matrix rows
    :param n_cols: int
        cooccurrence matrix columns
    :return csr_mat: csr_matrix
        sparse matrix
    """
    rows, cols, data = [], [], []
    for (row, col), value in dict.items():
        rows.append(row)
        cols.append(col),
        data.append(value)
    csr_mat = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return csr_mat