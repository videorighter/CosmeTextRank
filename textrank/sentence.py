# -*- coding: utf-8 -*-
"""
Author : videorighter
"""

from collections import Counter
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

from textrank.utils import vocab_indexing


def sent_graph(sents, tokenizer=None, min_count=2, min_sim=0.3, similarity=None, vocab2idx=None, verbose=False):
    """
     make sentence graph
    :param sents: list of str
        sentence list
    :param tokenize: callabe
        tokenizer return list of str
    :param min_count: int
        minimum term frequency
    :param min_sim: float
        minimum similarity between sentences
    :param similarity: callable or str
        similarity(s1, s2) returns float
        s1 and s2 are list of str
    :param vocab2idx: dict
        vocabulary to index mapper
        if None, this function scan vocabulary first
    :param verbose: boolean
        if True, verbose mode
    :return x: scipy.sparse.csr_matrix
        sentence similarity graph
        shape = (n sents, n sents)
    """

    if vocab2idx is None:
        idx2vocab, vocab2idx = vocab_indexing(sents, tokenizer, min_count)
    else:
        idx2vocab = [vocab for vocab, _ in sorted(vocab2idx.items(), key=lambda x: x[1])]

    x = sent2vec(sents, tokenizer, vocab2idx)
    if similarity == 'cosine':
        x = cosine_similarity_matrix
    return x


def sent2vec(sents, tokenizer, vocab2idx):
    """
    vectorize sentences
    :param sents: list of str
    :param tokenizer: callable
    :param vocab2idx: dict
        vocabulary to index mapper
    :return: scipy.sparse.csr_matrix
        shape=(n_rows, n_cols)
    """
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenizer(sent))  # 조건에 맞게 토크나이징 된 단어들의 수를 카운트
        for token, count in counter.items():
            j = vocab2idx.get(token, -1)
            if j == -1:  # default
                continue  # 만약 해당 token이 vocabulary에 없으면 넘김
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab2idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
    n_rows = x.shape[0]
    matrix = []
    for batch_idx in range(math.ceil(n_rows / batch_size)):  # 매트릭스 행을 배치사이즈로 나눈 값의 올림값 만큼 반복
        start = int(batch_idx * batch_size)
        end = min(n_rows, int((batch_idx + 1) * batch_size))
        pair_sim = 1 - pairwise_distances(x[start:end], x, metric='cosine')  # 코사인 유사도 거리값
        rows, cols = np.where(pair_sim >= min_sim)  # 코사인 유사도 거리값이 최소 유사도값보다 커질 경우의 인덱스
        data = pair_sim[rows, cols]
        matrix.append(csr_matrix((data, (rows, cols)), shape=(end-start, n_rows)))
        if verbose:
            print(f'\rcalculating cosine sentence similarity was done with {n_rows} sentences')
    return matrix


def textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()  # 0이 아닌 값들의 행 인덱스와 열 인덱스를 각각 array로 받음; (array, array)
    data = np.ones(rows.shape[0])  # 0이 아닌 값들의 개수
    csr_mat = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000  # 최소길이보다 작거나 같으면 10000
    size = np.log(size)

    matrix = []
    for batch_idx in range(math.ceil(n_rows/batch_size)):  # steps
        # slicing
        start = int(batch_idx * batch_size)
        end = min(n_rows, int((batch_idx+1) * batch_size))

        # dot product
        inner = csr_mat[start:end, :] * csr_mat.transpose()

        # sentence len[i, j] = size[i] + size[j]
        norm = size[start:end].reshape(-1, 1) + size.reshape(1, -1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0  # 정규화 과정에서 0으로 나눠진다면 0으로

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzereo()  # boolean matrix / 0이 아닌 경우 True
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        matrix.append(csr_matrix((data, (rows, cols)), shape=(end-start, n_rows)))  # 각 배치마다의 결과 append

        if verbose:
            print(f'\rcalculating textrank sentence similarity {start} / {n_rows}', end='')

    matrix = sp.sparse.vstack(matrix)
    if verbose:
        print(f'\rcalculating textrank sentence similarity was done with {n_rows} sentences')

    return matrix
