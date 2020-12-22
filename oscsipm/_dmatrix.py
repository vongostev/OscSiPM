# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:10:39 2019

@author: vonGostev
"""
import numpy as np
from scipy.special import binom
from scipy.linalg import pinv
from functools import lru_cache

from ._numpy_core import DPREC


def binomial_t_elem(qe: float, n: int, m: int):
    if m > n:
        return 0
    return qe ** m * (1 - qe) ** (n - m) * binom(n, m)


def binomial_t_matrix(qe: float, N: int, M: int):
    qe = DPREC(qe)
    t_matrix = np.zeros((M, N), dtype=DPREC)
    for n in range(N):
        for m in range(M):
            t_matrix[m, n] = binomial_t_elem(qe, n, m)
    return t_matrix


def binomial_invt_matrix(qe: float, N: int, M: int, n_cells=0):
    qe = DPREC(qe)
    t_matrix = np.zeros((N, M), dtype=DPREC)
    for n in range(N):
        for m in range(M):
            t_matrix[n, m] = binomial_t_elem(1 / qe, m, n)
    return t_matrix


@lru_cache(maxsize=1024)
def subbinom_t_elem(qe: float, n_cells: int, n: int, m: int):
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
    return (1 - qe + qe * m / n_cells) * subbinom_t_elem(qe, n_cells, n - 1, m) +\
        qe * (n_cells - m - 1) / n_cells * \
        subbinom_t_elem(qe, n_cells, n - 1, m - 1)


def subbinomial_t_matrix(qe: float, N: int, M: int, n_cells=667):
    qe = DPREC(qe)
    t_matrix = np.zeros((M, N), dtype=DPREC)
    for n in range(N):
        for m in range(M):
            t_matrix[m, n] = subbinom_t_elem(qe, n_cells, n, m)
    return t_matrix


def subbinomial_invt_matrix(qe: float, N: int, M: int, n_cells=667):
    return pinv(subbinomial_t_matrix(qe, N, M, n_cells))
