# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:51:49 2019

@author: Pavel Gostev
"""
import numpy as np

from scipy.optimize import brute, OptimizeResult
from .. import (g2, mean, normalize, P2Q, moment)
from ..stat import ppoisson
import logging


log = logging.getLogger('sc_crosstalk')
log.setLevel(logging.INFO)
info = log.info


def d_crosstalk_4n(p_crosstalk):
    """
    Probabilities of k ≤ 5 triggered pixels for the 4-neighbours model

    See table 1 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    ctnoise : ndarray
        Probability distributions of the total number
        of triggered pixels for a single initially fired pixel.
    """

    q_crosstalk = (1 - p_crosstalk)
    ctnoise = np.zeros(6)
    ctnoise[1] = q_crosstalk ** 4
    ctnoise[2] = 4 * p_crosstalk * q_crosstalk ** 6
    ctnoise[3] = 18 * p_crosstalk ** 2 * q_crosstalk ** 8
    ctnoise[4] = 4 * p_crosstalk ** 3 * q_crosstalk ** 8 * (
        1 + 3 * q_crosstalk + 18 * q_crosstalk ** 2)
    ctnoise[5] = 5 * p_crosstalk ** 4 * q_crosstalk ** 10 * (
        8 + 24 * q_crosstalk + 55 * q_crosstalk ** 2)
    return ctnoise


def p_crosstalk_m(m, k, p_crosstalk):
    """
    The probability of total number k of triggered pixels provided m primaries

    See formula 2.20 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    m : int
        Number of primary pixels.
    k : int
        Number of triggered pixels.
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    P_{m+1} : float
        The probability of total number k of triggered pixels
        provided m primaries

    """

    ctnoise = d_crosstalk_4n(p_crosstalk)
    if m == 1:
        if k > 5:
            return 0
        return ctnoise[k]
    return sum(p_crosstalk_m(m - 1, k - i, p_crosstalk) * ctnoise[i]
               for i in range(1, k - m + 2, 1) if i <= 5)


def p_crosstalk_m_pyomo(m, k, distr_crosstalk):
    """
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    Model with 4 neighbors with saturation
    See formula (14)
    """
    if m == 1:
        if k > 4:
            return 0
        return distr_crosstalk[k]
    return sum(p_crosstalk_m_pyomo(m - 1, k - i - 1, distr_crosstalk) * distr_crosstalk[i]
               for i in range(1, k - m + 2, 1) if i <= 4)


def distort(Qcorr, p_crosstalk):
    """
    Include crosstalk noise into photocounting statistics.
    We use model with 4 neighbors with saturation.

    See formula 2.19 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    Qcorr : iterable
        The photocounting statistics.
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    Q : ndarray
        Noised photocounting statistics.

    """

    N = len(Qcorr)
    Q = np.zeros(N)
    Q[0] = Qcorr[0]

    def genf(k):
        return sum(Qcorr[m] * p_crosstalk_m(m, k, p_crosstalk) for m in range(1, k + 1, 1))
    Q[1:] = [genf(k) for k in range(1, N, 1)]
    return Q


def compensate(Q, p_crosstalk):
    """
    Remove crosstalk noise from photocounting statistics.
    We use model with 4 neighbors with saturation.

    See formula 2.21 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    Q : iterable
        The photocounting statistics.
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    Qcorr : ndarray
        Denoised photocounting statistics.

    """
    N = len(Q)
    eps = total_pcrosstalk(p_crosstalk)
    Qcorr = np.zeros(N)
    Qcorr[0] = Q[0]
    Qcorr[1] = Q[1] / (1 - eps)
    for m in range(2, N, 1):
        Qcorr[m] = (1 - eps) ** -m * \
            (Q[m] - sum(Qcorr[k] * p_crosstalk_m(k, m, p_crosstalk)
                        for k in range(1, m, 1)))
    return normalize(np.abs(Qcorr))


def optctp(_pct_param, Q, PDE, N, mtype, n_cells):
    p_crosstalk, poisson_mean = _pct_param
    pm = P2Q(ppoisson(poisson_mean, N), PDE, len(Q),
             mtype=mtype, n_cells=n_cells)
    est = distort(pm, p_crosstalk)
    gdelta = (g2(est) - g2(Q)) ** 2
    return gdelta


def optimize_pcrosstalk(Q, PDE, N, mtype='binomial', n_cells=0, Ns=100):
    """
    Brute searching of crosstalk probability
    by optimizing of g2 difference from noised data and
    noised poisson photocounting statistics.
    We use model with 4 neighbors with saturation.

    See
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    Q : iterable
        Experimental photocounting statistics of a laser source.
    PDE : float
        PDE of the detector.
    N : int
        Size of poisson photon-number statistics.
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in the most of applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells in the subbinomial case. The default is 0.
    Ns : int, optional
        Size of calculation grid of brute function.
        The default is 100.

    Returns
    -------
    p_ct: float
        The probability of a single crosstalk event.

    res: OptimizeResult
        See scipy.optimize.brute description.
        res.x0 consists of optimal p_crosstalk and
        optimal poisson photon-number distribution
        p_crosstalk is in res.x0[0]

    """

    res = brute(optctp, ([0.00, 0.01], [mean(Q) / PDE * 0.9,
                                        mean(Q) / PDE * 1.1]),
                args=(Q, PDE, N, mtype, n_cells), Ns=Ns, full_output=True)
    info("P_ct = {r[0][0]}, Δg(2) = {r[1]}".format(r=res))
    return res[0][0], OptimizeResult(x=res[0], fval=res[1],
                                     grid=res[2], Jout=res[3])


def Q2total_pcrosstalk(Q):
    """
    Calculate model independent total crosstalk probability.
    It's may be vary from the result of optimize_pcrosstalk

    See formula 2.23 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    Q : iterable
        Experimental photocounting statistics of a pulsed laser source.

    Returns
    -------
    epsilon : float
        The total crosstalk probability.

    """

    mu = - np.log(Q[0])
    return 1 - Q[1] / mu / np.exp(- mu)


def total_pcrosstalk(p_crosstalk):
    """
    Calculate total crosstalk probability from
    the probability of single crosstalk event.
    We use model with 4 neighbors with saturation

    See text after formula 2.1 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    epsilon : float
        The total crosstalk probability.

    """

    return 1 - (1 - p_crosstalk)**4


def ENF(p_crosstalk):
    """
    Calculate excess noise factor (ENF) of the detector

    See formulas 2.16, 2.17 and 2.27 in
    Gallego, L., et al. "Modeling crosstalk in silicon photomultipliers."
    Journal of instrumentation 8.05 (2013): P05010.
    https://iopscience.iop.org/article/10.1088/1748-0221/8/05/P05010/pdf

    Parameters
    ----------
    p_crosstalk : float
        The probability of a single crosstalk event.

    Returns
    -------
    ENF : float
        Excess noise factor.

    """

    d = d_crosstalk_4n(p_crosstalk)
    r = d[5] / (1 - sum(d[1:5]))
    e1 = moment(d[:5], 1) + d[5] * (1 + 4*r) / r ** 2
    var1 = moment(d[:5], 2) + d[5] * (2 + 7*r + 16 * r ** 2) / r ** 3 - e1 ** 2
    return 1 + var1 / e1 ** 2
