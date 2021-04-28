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
import numba as nb

from compress_pickle import dump, load
from joblib import Parallel, delayed
from itertools import islice

gc.enable()

VENDOR_EXT = {'tek': 'wfm',
              'lecroy': 'trc'}


def parallelize(func, data, n_jobs, *func_args):
    return Parallel(n_jobs=n_jobs)(delayed(func)(d, *func_args) for d in data)


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


def parse_files(oscfiles, vendor, fsnum=0):
    if fsnum > 0:
        oscfiles = oscfiles[:fsnum]
    return [parse_file(df, vendor) for df in oscfiles]


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


@nb.jit(nopython=True, nogil=True, fastmath=True)
def single_pulse(osc: np.ndarray, method: str = 'max') -> np.float64:
    """
    Get amplitude of the single pulse

    Parameters
    ----------
    y : ndarray
        Oscillogram of the single pulse.

    method : str ('max', 'sum')

    """
    if method == 'sum':
        oscsum = 0
        for i in nb.prange(len(osc)):
            if osc[i] > 0:
                oscsum += osc[i]
        return oscsum
    elif method == 'max':
        return np.max(osc)


@np.vectorize
def random_pulse(osc, method='max'):
    # TODO: Make parameters external
    if method == 'sum':
        return np.sum(osc, dtype=float)
    elif method == 'max':
        y = osc[osc > 0]
        peaks = find_peaks(y, height=(0, None), distance=50,
                           width=(10, None))[1]['peak_heights']
        return np.sum(peaks)


@nb.jit(nopython=True, nogil=True, fastmath=True)
def _periodic_pulse(oscarray: np.ndarray, dx: float, frequency: float,
                    time_window: float, method: str = 'max') -> list:
    points_period = int(1 / frequency / dx) + 1
    points_window = int(time_window / dx) + 1

    init_point = np.argmax(oscarray)
    pulses_points = np.append(
        np.arange(init_point, 0, -points_period)[::-1],
        np.arange(init_point, len(oscarray), points_period))

    pulses = np.zeros(pulses_points.shape)

    for i in nb.prange(len(pulses_points)):
        p = pulses_points[i]
        if p < points_window:
            low = 0
            top = points_window
        else:
            low = p - points_window // 2
            top = p + points_window // 2
        pulses[i] = single_pulse(oscarray[low:top], method=method)

    return pulses


def periodic_pulse(data, frequency: float, time_window: float, method: str = 'max'):
    return _periodic_pulse(
        data.y, data.horizInterval, frequency, time_window, method)


def scope_unwindowed(data, time_discrete, method='max'):
    points_discrete = int(time_discrete // data.horizInterval) + 1
    split_indexes = np.arange(0, len(data.y), points_discrete)
    chunks = np.array(np.split(data.y.copy(), split_indexes), dtype='O')
    return random_pulse(chunks, method=method)


def from_memo(path, oscdata, vendor):
    _oscdata = parse_file(path, vendor)
    _oscdata.y = oscdata
    return _oscdata


def memo_oscillogram(data, vendor, correct_bs=True):
    path, oscdata = data

    if oscdata is None:
        oscdata = parse_file(path, vendor)
        y = oscdata.y
        y -= np.min(y)
        oscdata.y = correct_baseline(y) if correct_bs else y

    elif type(oscdata) == np.ndarray:
        oscdata = from_memo(path, oscdata, vendor)

    return (path, oscdata)


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

    disp: bool = False

    def __post_init__(self):
        if not self.parallel:
            self.parallel_jobs = 1
        if self.vendor not in self.vendors:
            raise ValueError('PulsesHistMaker.vendor must be in %s, not %s' %
                             (self.vendors, self.vendor))
        if self.method not in self.methods:
            raise ValueError('PulsesHistMaker.method must be in %s, not %s' %
                             (self.methods, self.method))
        self.rawdata = {}
        self.discretedata = {}

    def read(self, fsnum=0):
        if fsnum != 0:
            self.fsnum = fsnum

        datapaths = list_files(
            self.datadir, self.vendor, self.fsoffset, self.fsnum)

        # Check if file was processed
        files_processed = []

        for path in self.rawdata:
            if self.rawdata is not None:
                files_processed.append(path)

        for path in datapaths:
            if path not in files_processed:
                self.rawdata[path] = None

        if self.memo_file:
            with open(self.memo_file, 'rb') as f:
                self.rawdata.update(
                    load(f, compression='lzma',
                         set_default_extension=False))

        self.filesnum = len(self.rawdata) if not fsnum else fsnum

    def save_memo(self, filename):
        self.clear_rawdata()
        with open(filename, 'wb') as f:
            dump(dict(self.rawdata), f, compression='lzma',
                 set_default_extension=False)

    def save_hist(self, filename):
        np.savetxt(filename, np.vstack((self.bins[:-1], self.hist)).T)

    def clear_data(self):
        for k, d in self.rawdata.items():
            self.rawdata[k] = d.y
        self.discretedata = {}

    def single_pulse_hist(self, div_start=5.9, div_width=0.27):
        self.discretedata = {}
        for path, d in self.rawdata.items():
            x, y = windowed(d, div_start, div_width)
            self.discretedata.update({path: single_pulse(y, self.method)})
        self.make_hist(self.histbins)

    def multi_pulse_histogram(self, frequency=2.5e6, time_window=7.5e-9):
        self.parse(periodic_pulse, (frequency, time_window, self.method))
        self.make_hist(self.histbins)

    def unwindowed_histogram(self, time_discrete=15e-9):
        self.parse(scope_unwindowed, (time_discrete, self.method))
        self.make_hist(self.histbins)

    def parse(self, func, args):
        self.discretedata = {}

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)

            chunk_data = parallelize(
                memo_oscillogram, islice(self.rawdata.items(), i, hb),
                self.parallel_jobs, self.vendor, self.correct_baseline)
            self.rawdata.update(dict(chunk_data))

            chunk_pulses = parallelize(
                func, islice(self.rawdata.values(), i, hb),
                self.parallel_jobs, *args)
            self.discretedata.update(
                dict(zip(islice(self.rawdata.keys(), i, hb), chunk_pulses)))

            if self.disp:
                print(
                    f'Files ##{i + 1}-{hb + 1} T={time.time() - t:.2f} s', end='\t')

            del chunk_data
            del chunk_pulses
            gc.collect()

    def make_hist(self, histbins):
        pulses_data = np.concatenate(tuple(self.discretedata.values()))
        self.hist, self.bins = np.histogram(pulses_data, bins=histbins)

    def get_hist(self, histbins):
        self.make_hist(histbins)
        return self.bins[:-1], self.hist
