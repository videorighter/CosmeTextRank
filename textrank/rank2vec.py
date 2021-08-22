# -*- coding: utf-8 -*-
"""
Author : videorighter
"""

import numpy as np
from sklearn.preprocessing import normalize


def pagerank(x, bl_ratio, max_iter, bias=None):
    """
    Page Rank
    :param x: scipy.sparse.csr_matrix
        shape = (n_vertex, n_vertex)
    :param bl_ratio: float
        back link ratio, 0 < bl_ratio < 1
    :param max_iter: int
        maximum number of iteration
    :param bias: numpy.array or None
        if None, equal bias
    :return: numpy.ndarray
        page rank vector / shape = (n_vertex, 1)
    """

    assert 0 < bl_ratio < 1

    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)  # A의 row 길이의 vector를 transpose
    if bias is None:
        bias = (1 - bl_ratio) * np.ones(A.shape[0]).reshape(-1, 1)
    else:
        bias = bias.reshape(-1, 1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - bl_ratio) * bias

    # iteration
    for _ in range(max_iter):
        R = bl_ratio * (A * R) + bias

    return R