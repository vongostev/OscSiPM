# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:10:39 2019

@author: vonGostev
"""
import numpy as np
from scipy.special import comb
from scipy.linalg import pinv
from functools import lru_cache

from ._numpy_core import DPREC


def d_binomial(qe: float, N: int, M: int):
    m = np.arange(M).reshape((-1, 1))
    n = np.arange(N).reshape((1, -1))
    return comb(n, m, exact=False) * qe ** m * (1 - qe)**(n - m)


def invd_binomial(qe: float, N: int, M: int, n_cells: int = 0):
    return d_binomial(1 / qe, M, N)


@lru_cache(maxsize=None)
def sb_elem(qe: float, n_cells: int, n: int, m: int):
    """
    Calculated from formula (5) via recursive approach from Appendix A

    Sperling, J., Vogel, W., & Agarwal, G. S. (2014).
    Quantum state engineering by click counting.
    Physical Review A, 89(4), 043829.
    for arbitrary quantum efficiency and zero noise

    Idea from
    Sperling, J., Vogel, W., & Agarwal, G. S. (2012).
    True photocounting statistics of multiple on-off detectors.
    Physical Review A, 85(2), 023820.

    """
    if n == 0 and m == 0:
        return 1
    elif n != 0 and m == 0:
        return (1 - qe) ** n
    elif m > n:
        return 0
    return (1 - qe + qe * m / n_cells) * sb_elem(qe, n_cells, n - 1, m) +\
        qe * (n_cells - m - 1) / n_cells * sb_elem(qe, n_cells, n - 1, m - 1)


def d_subbinomial(qe: float, N: int, M: int, n_cells: int = 1000):
    qe = DPREC(qe)
    m = np.arange(M).reshape((-1, 1))
    n = np.arange(N).reshape((1, -1))
    d_matrix = np.vectorize(sb_elem, otypes=[DPREC])(qe, n_cells, n, m)
    return d_matrix


def invd_subbinomial(qe: float, N: int, M: int, n_cells: int = 1000):
    return pinv(d_subbinomial(qe, N, M, n_cells))
