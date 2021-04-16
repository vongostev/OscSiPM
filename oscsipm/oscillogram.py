# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:25:54 2019

@author: Pavel Gostev
"""
from os.path import isfile, join
from os import listdir
import time

import gc

from dataclasses import dataclass

import lecroyparser
import tekwfm2 as tek

from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.signal import find_peaks

import numpy as np

from compress_pickle import dump, load
from joblib import Parallel, delayed

gc.enable()

VENDOR_EXT = {'tek': 'wfm',
              'lecroy': 'trc'}


def parse_file(datafile, vendor):
    if vendor == 'lecroy':
        data = lecroyparser.ScopeData(datafile)
        delattr(data, "file")
    if vendor == 'tek':
        data = tek.ScopeData(datafile)
    return data


def list_files(datadir, vendor, fsoffset, fsnum):
    lf = [join(datadir, f) for f in listdir(datadir)
          if isfile(join(datadir, f)) and f.endswith(VENDOR_EXT[vendor])]
    if fsoffset != 0:
        lf = lf[fsoffset:]
    if fsnum > 0:
        lf = lf[:fsnum]
    return lf


def parse_files(oscfiles, vendor, fsnum=0, parallel=False):
    n_jobs = -1 if parallel else 1

    if fsnum > 0:
        oscfiles = oscfiles[:fsnum]
    return Parallel(
        n_jobs=n_jobs)([delayed(parse_file)(df, vendor) for df in oscfiles])


def windowed(data, div_start, div_width):
    div_points = len(data.x) / data.horizInterval
    wstart = int(div_start * div_points)
    wwidth = int(div_width * div_points)
    return data.x[wstart:wstart + wwidth], data.y[wstart:wstart + wwidth]


def baseline_als(y, lam, p, niter=2):
    """
    See "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005
    https://www.researchgate.net/publication/228961729_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def correct_baseline(y, lam=1e5, p=0.1):
    baseline_values = baseline_als(y, lam=lam, p=p)
    return y - baseline_values


def single_pulse(y, method='max'):
    """
    Get amplitude of the single pulse

    Parameters
    ----------
    y : ndarray
        Oscillogram of the single pulse.

    method : str ('max', 'sum')

    """
    if method == 'sum':
        return np.sum(y[y >= 0], dtype=float)
    elif method == 'max':
        return max(y)


def random_pulse(y, method='max'):
    if method == 'sum':
        return np.sum(y, dtype=float)
    elif method == 'max':
        peaks, _ = find_peaks(y[y > 0], distance=50, width=10)
        return sum(y[y > 0][peaks])


def periodic_pulse(data, frequency, time_window, method='max'):
    discretedata = []
    points_period = int(1 / frequency / data.horizInterval) + 1
    points_window = int(time_window / data.horizInterval) + 1
    y = data.y
    init_point = np.argmax(y)
    pulses_points = np.append(
        np.arange(init_point, 0, -points_period)[::-1],
        np.arange(init_point, len(y), points_period))
    for p in pulses_points:
        if p < points_window:
            low = 0
            top = points_window
        else:
            low = p - points_window // 2
            top = p + points_window // 2
        discretedata.append(single_pulse(y[low:top], method=method))
    return discretedata


def scope_unwindowed(data, time_discrete, method='max'):
    points_discrete = int(time_discrete // data.horizInterval)
    y = data.y
    points_discrete += 1
    discretedata = [random_pulse(y[i:i + points_discrete], method=method)
                    for i in range(0, len(y), points_discrete)]
    return discretedata


def memo_oscillogram(data, vendor, correct_bs=True):
    if type(data) is str:
        filedata = (data, parse_file(data, vendor))
    if type(data) == tuple:
        if type(data[1]) is not str:
            return data

        filedata = (data[0], parse_file(data[0], vendor))
        filedata[1].y = data[1]
        return filedata

    y = filedata[1].y
    y -= np.min(y)
    filedata[1].y = correct_baseline(y) if correct_bs else y

    return filedata


@dataclass
class PulsesHistMaker:
    """
    Class for extract amplitude histogram from plenty of oscillogram files
    Supported formats are:
        - LecCroy binary .trc
        - Tektronix binary v.2 .wfm

    __init__ arguments
    ------------------

    datadir: str
        Path to oscillogram files
    fsnum: int = -1
        Number of files to process.
        If -1 all files in the datadir will be processed
    fsoffset: int = 0
        Skip files from the first ine in the datadir
    fchunksize: int = 10
        The size of chunk for parallel processing
    parallel: bool = False
        Use joblib or not?
    parallel_jobs: int = -1
        Number of parallel jobs
    memo_file: str = ''
        Path to file with compressed data from previous processing.
        Can be use iff use save it previously with 'save_memo' method
    histbins: int = 2000
        Numbere of bins in the histogram.
        Can be change in methods 'make_hist' and 'get_hist'
    correct_baseline: bool = True
        Correct baseline of not?
    method : str = 'max'
        Method of pulses counting. Can be 'max' or 'sum'
    vendor: str = 'lecroy'
        Vendor name. Can be 'tek' or 'lecroy'
    """

    datadir: str
    fsnum: int = -1
    fsoffset: int = 0
    fchunksize: int = 10
    parallel: bool = False
    parallel_jobs: int = -1
    memo_file: str = ''
    histbins: int = 2000
    correct_baseline: bool = True

    method: str = 'max'
    methods: tuple = ('sum', 'max')

    vendor: str = 'lecroy'
    vendors: tuple = ('lecroy', 'tek')

    def __post_init__(self):
        if not self.parallel:
            self.parallel_jobs = 1
        if self.vendor not in self.vendors:
            raise ValueError('vendor must be in %s, not %s' %
                             (self.vendors, self.vendor))
        if self.method not in self.methods:
            raise ValueError('method must be in %s, not %s' %
                             (self.methods, self.method))

    def read(self, fsnum=-1, parallel_read=False):
        if fsnum == -1:
            fsnum = self.fsnum
        self.rawdata = list_files(
            self.datadir, self.vendor, self.fsoffset, fsnum)
        if self.memo_file:
            with open(self.memo_file, 'rb') as f:
                memodata = load(f, compression='lzma',
                                set_default_extension=False)
            for k, path in enumerate(self.rawdata):
                if path in self.rawdata:
                    self.rawdata[k] = (path, memodata[path])

        self.filesnum = len(self.rawdata)

    def save_memo(self, filename):
        self.clear_rawdata()
        with open(filename, 'wb') as f:
            dump(dict(self.rawdata), f, compression='lzma',
                 set_default_extension=False)

    def save_hist(self, filename):
        np.savetxt(filename, np.vstack((self.bins[:-1], self.hist)).T)

    def clear_rawdata(self):
        for k in range(len(self.rawdata)):
            p, d = self.rawdata[k]
            self.rawdata[k] = (p, d.y)

    def single_pulse_hist(self, div_start=5.9, div_width=0.27):
        self.discretedata = []
        i = 1
        for d in self.rawdata:
            x, y = windowed(d, div_start, div_width)
            self.discretedata.append(single_pulse(y, self.method))
            i += 1
        self.make_hist(self.histbins)

    def multi_pulse_histogram(self, frequency=2.5e6, time_window=7.5e-9):
        self.parse(periodic_pulse, (frequency, time_window, self.method))
        self.make_hist(self.histbins)

    def unwindowed_histogram(self, time_discrete=15e-9):
        self.parse(scope_unwindowed, (time_discrete, self.method))
        self.make_hist(self.histbins)

    def parse(self, func, args):
        self.discretedata = []

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)
            self.rawdata[i:hb] = Parallel(
                n_jobs=self.parallel_jobs)(
                    delayed(memo_oscillogram)(
                        df, self.vendor, self.correct_baseline)
                for df in self.rawdata[i:hb])
            pulsesdata = [func(df[1], *args) for df in self.rawdata[i:hb]]
            self.discretedata += pulsesdata

            print('Files ##%d-%d time %.2f s' %
                  (i, hb, time.time() - t), end='\t')

            del pulsesdata
            gc.collect()

    def make_hist(self, histbins):
        self.hist, self.bins = np.histogram(self.discretedata, bins=histbins)

    def get_hist(self, histbins):
        self.make_hist(histbins)
        return self.bins[:-1], self.hist
