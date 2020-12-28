# -*- coding: utf-8 -*-
"""
@author: Pavel Gostev
"""
from ._numpy_core import normalize, lrange, abssum

import numpy as np
from scipy.signal import find_peaks
from scipy.special import eval_hermitenorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def loadhist(path, **kwargs):
    for dl in [',', ' ', '\t']:
        try:
            bins, hist = np.loadtxt(path, delimiter=dl, unpack=True, **kwargs)
        except (ValueError, TypeError, FileNotFoundError, IOError) as E:
            np.warnings.warn_explicit(
                'Histogram parsing with delimiter' + r' "%s": %s' % (dl, E),
                IOError, __file__, 18)
            continue
        else:
            return bins, hist

    raise ValueError("Can't parse the histogram from the file %s" % path)


def gauss_hermite_poly(x, norm_factor, peak_pos, sigma, h3, h4):
    w = (x - peak_pos) / sigma
    return norm_factor * np.exp(- w ** 2 / 2) * (1 + h3*eval_hermitenorm(3, w) + h4*eval_hermitenorm(4, w))


def peak_area(norm_factor, peak_pos, sigma, h3, h4):
    return norm_factor * sigma * (np.sqrt(2*np.pi) + h4)


def minpoly(popt, bins, hist):
    return np.linalg.norm(gauss_hermite_poly(bins, *popt) - hist, 1)


def mindowns(popt, bins, hist, downs):
    return np.linalg.norm((hist - gauss_hermite_poly(bins, *popt))[downs])


def hist2Q(hist, bins, discrete,
           threshold=1, peak_width=1,
           plot=False, method='sum',
           remove_pedestal=1,
           lim_downssum=1,
           maxiter=20):
    """
    Build photocounting statistics from an experimental histogram
    by gaussian-hermite polynoms or simple sum

    Parameters
    ----------
    hist : iterable
        The experimental histogram.
    discrete : int
        The amplitude of single photocount pulse in points.
    threshold : int, optional
        Minimal number of events to find histogram peak.
        The default is 1.
    peak_width : int, optional
        The width of peaks.
        It must be 1 if the histogram is made by 'count' method.
        It must be greater if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.
    method : {'sum', 'fit'}
        Method of the photocounting statistics construction.
            'sum' is a simple summation between minimums of the histogram

            'fit' is a gauss-hermite function fitteing like in [1]
    remove_pedestal : boolean
        Flag to remove pedestal noise from the histogram
    lim_downssum : float
        Limit for sum of downs of the histogram
    maxiter : int
        Maximum iterations count for pedestal removing
        
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
    downs, _ = find_peaks(-hist, distance=discrete, width=1)
    if plot:
        plt.plot(bins, hist)
    if remove_pedestal:
        it = 0
        while sum(hist[downs]) > lim_downssum and it < maxiter:
            dres = minimize(mindowns, args=(bins, hist, downs),
                            x0=(1, bins[len(bins) // 2], np.sqrt(bins[len(bins) // 2]), 0, 0))
            hist -= gauss_hermite_poly(bins, *dres.x)
            hist[hist < 0] = 0
            downs, _ = find_peaks(-hist, distance=discrete, width=1)
            it += 1
    
        if plot:
            plt.plot(bins, hist)

    peaks, _ = find_peaks(hist, threshold=threshold, distance=discrete,
                          width=peak_width)
    if any(bins < 0):
        firstdown = min(np.argmin(hist[bins < 0]),
                        downs[0] - int(discrete * 1.2))
    else:
        firstdown = 0
    downs = np.sort(np.append([firstdown], downs))
    if plot:
        plt.scatter(bins[peaks], hist[peaks])
        plt.scatter(bins[downs], hist[downs])
    plt.show()

    Q = []
    for i in lrange(downs)[:-1]:
        for p in peaks:
            dl = downs[i]
            dt = downs[i+1]
            if p < dl or p > dt:
                continue
            if method == 'sum':
                Q.append(sum(hist[dl:dt]))
            if method == 'fit':
                res = minimize(minpoly, args=(bins[dl:dt], hist[dl:dt]),
                               x0=(hist[p], bins[p], bins[1] - bins[0], 0.0, 0.0),
                               bounds=list(zip(
                                   [hist[p] / 2, bins[dl], bins[1] -
                                       bins[0], -0.01, -0.01],
                                   [hist[p], bins[dt], np.sqrt(bins[dt] - bins[dl]) / 6, 0.01, 0.01])))
                popt = res.x
                Q.append(peak_area(*popt))
                if plot:
                    plt.plot(bins, gauss_hermite_poly(bins, *popt))
            break

    if plot and method == 'fit':
        plt.plot(bins, hist)
        plt.show()
    return normalize(Q)


class QStatisticsMaker:
    """
    Class to make photocounting statistics from histogram

    __init__ arguments
    ----------
    fname : string
        File name contains the histogram.
    photon_discrete : float
        The amplitude of the single-photocount pulse.
    peak_width : int, optional
        The width of peaks.
        It must be 1 if the histogram is made by 'count' method.
        It must be greater if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    method: {'sum', 'fit'}
        Method of the photocounting statistics construction.
            'sum' is a simple summation between minimums of the histogram

            'fit' is a gauss-hermite function fitting like in [1]
    skiprows : int, optional
        Number of preamble rows in the file. The default is 0.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.
    remove_pedestal : boolean
        Flag to remove pedestal from the histogram
        
    References
    ----------
    .. [1]
    Ramilli, Marco, et al. "Photon-number statistics with silicon photomultipliers."
    JOSA B 27.5 (2010): 852-862.

    """

    def __init__(self, fname, photon_discrete,
                 peak_width=1, method='fit', skiprows=0, plot=False,
                 remove_pedestal=True):
        self.photon_discrete = photon_discrete
        self.fname = fname
        self.plot = plot

        self._extract_data(skiprows)
        self.Q = hist2Q(self.hist, self.bins, discrete=self.points_discrete // 1.2,
                        peak_width=peak_width, plot=self.plot, method=method,
                        remove_pedestal=remove_pedestal)

    # Reading information from the file
    def _extract_data(self, skiprows):
        self.bins, self.hist = loadhist(self.fname, skiprows=skiprows)
        if self.bins[1] - self.bins[0] < 0:
            self.bins = self.bins[::-1]
            self.hist = self.hist[::-1]
        self.points_discrete = int(self.photon_discrete /
                                   (self.bins[1] - self.bins[0]))

    def getq(self):
        """
        Returns the photocounting statistics was made

        Returns
        -------
        self.Q : ndarray
            self.Q[self.Q > 0].

        """

        self.Q = self.Q[self.Q > 0]
        return normalize(self.Q)
