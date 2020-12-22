# -*- coding: utf-8 -*-
"""
Created on Fri Oct 02 08:39:31 2019

@author: Pavel Gostev
@email: gostev.pavel@physics.msu.ru

Версия алгоритма расчета статистики фотоотсчетов
на основе numpy.float128
"""

import numpy as np

from ._numpy_core import normalize
from ._dmatrix import binomial_t_matrix, subbinomial_t_matrix
from ._dmatrix import binomial_invt_matrix, subbinomial_invt_matrix


def t_matrix(qe: float, N: int, M: int, mtype='binomial', n_cells=0):
    """
    Method for construction of binomial or subbinomial photodetection matrix
    with size NxM

    Parameters
    ----------
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int
        Maximum numbers of photons
    M : int
        Maximum number of photocounts
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in the most of applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells in the subbinomial case. The default is 0.

    Raises
    ------
    ValueError
        Wrong method for matrix construction.
        mtype must be binomial or subbinomial.

    Returns
    -------
    numpy.ndarray
        Binomial or subbinomial photodetection matrix of size NxM.

    """

    if mtype == 'binomial':
        return binomial_t_matrix(qe, N, M)
    elif mtype == 'subbinomial':
        return subbinomial_t_matrix(qe, N, M, n_cells)
    else:
        raise ValueError("""
                         Can't construct detection matrix of type %s
                         mtype must be binomial or subbinomial
                         """ % mtype)


def invt_matrix(qe, N, M, mtype='binomial', n_cells=0):
    """
    Method for construction of binomial or
    subbinomial inverse photodetection matrix
    with size MxN

    Parameters
    ----------
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int
        Maximum numbers of photons
    M : int
        Maximum number of photocounts
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Raises
    ------
    ValueError
        Wrong method for matrix construction.
        mtype must be binomial or subbinomial..

    Returns
    -------
    numpy.ndarray
        Binomial or subbinomial inverse photodetection matrix of size MxN.

    """
    if mtype == 'binomial':
        return binomial_invt_matrix(qe, N, M)
    elif mtype == 'subbinomial':
        return subbinomial_invt_matrix(qe, N, M, n_cells)
    else:
        raise ValueError("""
                         Can't construct inversed detection matrix of type %s
                         mtype must be binomial or subbinomial
                         """ % mtype)


def P2Q(P: np.ndarray, qe: float, M=0, mtype='binomial', n_cells=0):
    """
    Method for calculation of photocounting statistics
    from photon-number statistics

    Parameters
    ----------
    P : numpy.ndarray
        Photon-number statistics.
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    M : int, optional
        Maximum number of photocounts. It's undependent from length of P
        The default is 0, and in this case length(Q) = length(P).
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Returns
    -------
    Q : numpy.ndarray
        Photocounting statistics.

    """

    N = len(P)
    if M == 0:
        M = N
    return normalize(t_matrix(qe, N, M, mtype, n_cells).dot(P))


def Q2P(Q: np.ndarray, qe: float, N=0, mtype='binomial', n_cells=0):
    """
    Method for calculation of photon-number statistics
    from photocounting statistics by simple inversion

    Parameters
    ----------
    Q : numpy.ndarray
        Photocounting statistics.
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int, optional
        Maximum number of photons. It's undependent from length of Q
        The default is 0, and in this case length(Q) = length(P).
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Returns
    -------
    P : numpy.ndarray
        Photon-number statistics.

    """
    M = len(Q)
    if N == 0:
        N = M
    if N > M:
        Q = np.concatenate((Q, np.zeros(N - M)))
        M = N

    return normalize(invt_matrix(qe, M, N, mtype, n_cells).dot(Q))
