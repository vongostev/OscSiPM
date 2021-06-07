# -*- coding: utf-8 -*-
"""
@author: Pavel Gostev
"""
from dataclasses import dataclass

from fpdet import normalize, lrange

import numpy as np
from scipy.signal import find_peaks
from scipy.special import eval_hermitenorm as eval_hn
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def loadhist(path: str, skiprows: int = 0):
    for dl in [',', ' ', '\t']:
        try:
            bins, hist = np.loadtxt(
                path, delimiter=dl, unpack=True, skiprows=skiprows)
        except (ValueError, TypeError, IOError):
            continue
        except FileNotFoundError as E:
            raise FileNotFoundError(E)
        else:
            print(f'Histogram file {path} parsed with delimiter {dl}')
            return bins, hist

    raise ValueError("Can't parse the histogram from the file %s" % path)


def gauss_hermite_poly(x: float, norm_factor: float, peak_pos: float,
                       sigma: float, h3: float, h4: float) -> float:
    w = (x - peak_pos) / sigma
    gaussian = norm_factor * np.exp(- w ** 2 / 2)
    return gaussian * (1 + h3*eval_hn(3, w) + h4*eval_hn(4, w))


def peak_area(norm_factor: float, peak_pos: float, sigma: float,
              h3: float, h4: float) -> float:
    return norm_factor * sigma * (np.sqrt(2*np.pi) + h4)


def minpoly(popt: np.ndarray,
            hist: np.ndarray, bins: np.ndarray) -> float:
    return np.linalg.norm(gauss_hermite_poly(bins, *popt) - hist, 1)


def mindowns(popt: np.ndarray,
             hist: np.ndarray, bins: np.ndarray, downs: np.ndarray) -> float:
    return np.linalg.norm((hist - gauss_hermite_poly(bins, *popt))[downs])


def construct_q_sum(hist: np.ndarray,
                    peaks: np.ndarray,
                    downs: np.ndarray) -> np.ndarray:
    Q = []
    for i in lrange(downs)[:-1]:
        for p in peaks:
            dl = downs[i]
            dt = downs[i+1]
            if p >= dl and p <= dt:
                Q.append(sum(hist[dl:dt]))
                break
    return Q


def construct_q_fit(hist: np.ndarray, bins: np.ndarray,
                    peaks: np.ndarray, downs: np.ndarray) -> np.ndarray:
    Q = []
    for i in lrange(downs)[:-1]:
        for p in peaks:
            dl = downs[i]
            dt = downs[i+1]
            if p >= dl and p <= dt:
                fit_bounds = [[hist[p] / 2, hist[p]],
                              [bins[dl], bins[dt]],
                              [bins[1] - bins[0],
                                  np.sqrt(bins[dt] - bins[dl]) / 8],
                              [-0.01, 0.01],
                              [-0.01, 0.01]]
                x0 = (hist[p], bins[p], bins[1] - bins[0], 0.0, 0.0)
                res = minimize(minpoly, args=(hist[dl:dt], bins[dl:dt]),
                               x0=x0, bounds=fit_bounds)
                popt = res.x
                Q.append(peak_area(*popt))
                break
    return Q


def hist2Q(hist: np.ndarray, bins: np.ndarray, discrete: int,
           method: str = 'sum', threshold: float = 1,
           peak_width: float = 1, down_width: float = 1,
           manual_zero_offset: int = 0,
           plot: bool = False, logplot: bool = False, *args, **kwargs) -> np.ndarray:
    """
    Build photocounting statistics from an experimental histogram
    by gaussian-hermite polynoms or simple sum

    Parameters
    ----------
    hist : ndarray
        Experimental histogram values.
    bins : ndarray
        Experimental histogram bins.
    discrete : int
        The amplitude of single photocount pulse in points.
    manual_zero_offset: int, optional
        Offset for calculate zero-photon probability in presence of non-zero noise pulses (in points).
        The default is 0.
    threshold : float, optional
        Minimal number of events to find histogram peak.
        The default is 1.
    peak_width : float, optional
        The width of peaks.
        It must be greater than 1 if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    down_width : float, optional
        The width of downs.
        It must be greater than 1 if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.
    logplot : bool
        Enable log yscale for histogram plotting.
        The default is False.        
    method : {'sum', 'fit', 'manual'}
        Method of the photocounting statistics construction.
            'sum' is a simple summation between minimums of the histogram

            'fit' is a gauss-hermite function fitteing like in [1]

            'manual' is a simple summation of intervals with fixed length. 

    Returns
    -------
    Q : ndarray
        The photocounting statistics.

    References
    ----------
    .. [1]
    Ramilli, Marco, et al. "Photon-number statistics with silicon photomultipliers."
    JOSA B 27.5 (2010): 852-862.

    """
    if method != 'manual':
        discrete = int(discrete * 0.9)
        downs, _ = find_peaks(-hist, distance=discrete, width=down_width)
        downs = np.append([0], downs)

        peaks, _ = find_peaks(np.concatenate(([0], hist)), threshold=threshold, distance=discrete,
                              width=peak_width, plateau_size=(0, 10))
        peaks -= 1

        if peaks == []:
            raise ValueError(
                'Histogram peaks were not found with given settings')

        if plot:
            plt.scatter(bins[peaks], hist[peaks])
            plt.scatter(bins[downs], hist[downs])

    if method == 'manual':
        Q = []
        peaks = np.arange(manual_zero_offset, len(bins), discrete)

        for p in peaks:
            if p == manual_zero_offset:
                low = 0
            else:
                low = int(max(0, p - discrete // 2))
            top = int(min(p + discrete // 2, len(hist) - 1))
            Q.append(np.sum(hist[low:top]))

            if plot:
                plt.axvline(bins[low], linestyle=':', color='black')
                plt.axvline(bins[top], linestyle=':', color='black')

        if top != len(hist) - 1:
            Q.append(sum(hist[top:]))

    elif method == 'sum':
        Q = construct_q_sum(hist, peaks, downs)
    elif method == 'fit':
        Q = construct_q_fit(hist, bins, peaks, downs)

    if plot:
        plt.plot(bins, hist)
        plt.xlabel('Amplitude, V')
        plt.ylabel("Events' number")
        if logplot:
            plt.yscale('log')
        plt.show()

    return normalize(Q)


@dataclass
class QStatisticsMaker:
    """
    Class to make photocounting frequency distribution (statistics) from the histogram file.
    A format of the file must be texted with lines organized as follows:
        bin_value<delimiter from {',', ' ', '\t'}>hist_value\n

    One can use three different methods to construct frequency distribution: ('sum', 'fit', 'manual').
    The best tested one is 'manual'. Two others are experimental and can be unstable.

    Parameters
    ----------
    filename : string
        File name contains the histogram.
    amplitude_discrete : float
        The amplitude of single-photocount pulses.
    amplitude_zero_offset: float, optional
        Offset for calculate zero-photon probability in presence of non-zero noise pulses.
        The default is 0.
    peak_width : int, optional
        The width of peaks.
        It must be 1 if the histogram is made by 'count' method.
        It must be greater if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    down_width : float, optional
        The width of downs.
        It must be greater than 1 if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.
    logplot : bool
        Enable log yscale for histogram plotting.
        The default is False.        
    method : ('sum', 'fit', 'manual')
        Method of the photocounting statistics construction.
            - 'sum' is a simple summation between minimums of the histogram
            - 'fit' is a gauss-hermite function fitteing like in [1]
            - 'manual' is a simple summation of intervals with fixed length
    skiprows : int, optional
        Number of preamble rows in the file. The default is 0.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.
    logplot : bool
        Enable log yscale for histogram plotting.
        The default is False.

    References
    ----------
    .. [1]
    Ramilli, Marco, et al. "Photon-number statistics with silicon photomultipliers."
    JOSA B 27.5 (2010): 852-862.

    """
    filename: str
    amplitude_discrete: float
    amplitude_zero_offset: float = 0
    method: str = 'manual'
    methods: tuple = ('sum', 'fit', 'manual')

    skiprows: int = 0

    peak_width: float = 1.
    down_width: float = 1.

    plot: bool = False
    logplot: bool = False

    def __post_init__(self):

        if self.method not in self.methods:
            raise ValueError(
                f"{self.__name__}.method must be in {self.methods}, not {self.method}")

        self.bins, self.hist = loadhist(self.filename, skiprows=self.skiprows)
        # Cut bins and hist if manual method is used
        # Inverse data if bins are ordered descending
        if self.bins[1] < self.bins[0]:
            self.bins = self.bins[::-1]
            self.hist = self.hist[::-1]

        if self.method == self.methods[2]:
            neg_counts = np.sum(self.hist[self.bins < 0])
            self.hist = self.hist[self.bins >= 0]
            self.bins = self.bins[self.bins >= 0]
            self.hist[0] += neg_counts

        dx = self.bins[1] - self.bins[0]
        self.manual_zero_offset = int(self.amplitude_zero_offset / dx)
        self.discrete = int(self.amplitude_discrete / dx)
        self._q = hist2Q(**self.__dict__)

    @property
    def Q(self):
        """
        Returns the photocounting statistics was made

        Returns
        -------
        self.Q : ndarray

        """

        return self._q
