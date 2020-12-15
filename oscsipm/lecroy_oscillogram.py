# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:25:54 2019

@author: Pavel Gostev
"""
import gc
from dataclasses import dataclass
from scipy.signal import find_peaks

import lecroyparser
from . import tekwfm as tek

from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
from os.path import isfile, join
from os import listdir
import time
from compress_pickle import dump, load
from joblib import Parallel, delayed

gc.enable()


def parse_file(datafile, vendor):
    if vendor == 'lecroy':
        data = lecroyparser.ScopeData(datafile)
        delattr(data, "file")
    if vendor == 'tek':
        data = tek.ScopeData(datafile)
    return data


def list_files(datadir, fsoffset, fsnum):
    lf = [join(datadir, f) for f in listdir(datadir)
            if isfile(join(datadir, f))]
    if fsoffset != 0:
        lf = lf[fsoffset:]
    if fsnum > 0:
        lf = lf[:fsnum]
    return lf


def parse_files(oscfiles, vendor, fsnum=0, parallel=False):

    if fsnum > 0:
        oscfiles = oscfiles[:fsnum]
    if parallel:
        data = Parallel(n_jobs=-1)([delayed(parse_file)(df, vendor) for df in oscfiles])
    else:
        data = [parse_file(df, vendor) for df in oscfiles]
    return data


def windowed(data, div_start, div_width):
    div_points = len(data.x) / data.horizInterval
    wstart = int(div_start * div_points)
    wwidth = int(div_width * div_points)
    return data.x[wstart:wstart + wwidth], data.y[wstart:wstart + wwidth]


def baseline_als(y, lam, p, niter=2):
    # "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005
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


def single_pulse(y, discrete, method='max'):
    """
    Get amplitude of the single pulse

    Parameters
    ----------
    y : ndarray
        Oscillogram of the single pulse.

    Returns
    -------
    Amplitude : float
        Amplitude of the pulse.
        It can be discrete (if 'counts')
        or continuous (if 'max')

    """

    if method == 'counts':
        peaks = find_peaks(y, 0.02, 100)
        ypeaks = y[peaks]
        count = 0
        for p in ypeaks:
            for m in range(100):
                if abs(p / discrete - m) < 0.25:
                    count += m
                    break
        return count
    elif method == 'max':
        return max(y)


def periodic_pulse(data, frequency, time_window, discrete, method='max'):
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
        discretedata.append(single_pulse(y[low:top], discrete, method))
    return discretedata


def memo_oscillogram(data, vendor, correct_bs=True):
    if type(data) is str:
        filedata = (data, parse_file(data, vendor))
    if type(data) == tuple:
        if type(data[1]) is not str:
            return filedata
        filedata = (data[0], parse_file(data[0], vendor))
        filedata[1].y = data[1]
        return filedata

    y = filedata[1].y
    y -= np.min(y)
    if correct_bs:
        filedata[1].y = correct_baseline(y)
    else:
        filedata[1].y = y
    return filedata

@dataclass
class PulsesHistMaker:
    datadir: str
    method: str = 'max'
    discrete: float = 1
    fsnum: int = -1
    fsoffset: int = 0
    fchunksize: int = 10
    parallel: bool = False
    parallel_jobs: int = -1
    memo_file: str = ''
    histbins: int = 2000
    correct_baseline: bool = True
    vendor : str = 'lecroy'
    
    methods : tuple = ('max', 'counts')
    vendors : tuple = ('lecroy', 'tek')

    def __post_init__(self):
        if not self.parallel:
            self.parallel_jobs = 1
        if self.method not in self.methods:
            raise ValueError('method must be in %s, not %s' %
                             (self.methods, self.method))
        if self.vendor not in self.vendors:
            raise ValueError('vendor must be in %s, not %s' %
                             (self.vendors, self.vendor))
            
    def read(self, fsnum=-1, parallel_read=False):
        if fsnum == -1:
            fsnum = self.fsnum
        self.rawdata = list_files(self.datadir, self.fsoffset, fsnum)
        if self.memo_file:
            with open(self.memo_file, 'rb') as f:
                memodata = load(f, compression='lzma', set_default_extension=False)
            for r in self.rawdata:
                if r in memodata:
                    r = (r, memodata[r])
        self.filesnum = len(self.rawdata)
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            dump(self.rawdata, f, compression='lzma', set_default_extension=False)
            
    def save_hist(self, filename):
        np.savetxt(filename, np.vstack((self.bins[:-1], self.hist)).T)

    def clear_rawdata(self, i, hb):
        for k in range(i, hb):
            p, d = self.rawdata[k]
            self.rawdata[k] = (p, d.y)
            
    def single_pulse_hist(self, div_start=5.9, div_width=0.27):
        discretedata = []
        i = 1
        for d in self.rawdata:
            x, y = windowed(d, div_start, div_width)
            discretedata.append(single_pulse(y, self.discrete, self.method))
            i += 1
        self.hist, self.bins = np.histogram(discretedata, bins=self.histbins)

    def multi_pulse_histogram(self, frequency=2.5e6, time_window=7.5e-9):
        self.parse(periodic_pulse, (frequency, time_window, self.discrete, self.method))

    def scope_unwindowed(self, data, time_discrete):
        points_discrete = int(time_discrete // data.horizInterval)
        y = data.y
        points_discrete += 1
        discretedata = [single_pulse(y[i:i + points_discrete], self.discrete, self.method)
                        for i in range(0, len(y), points_discrete)]
        return discretedata

    def unwindowed_histogram(self, time_discrete=15e-9):
        self.parse(self.scope_unwindowed, (time_discrete,))

    def parse(self, func, args):
        discretedata = []

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)
            self.rawdata[i:hb] = Parallel(n_jobs=self.parallel_jobs)([
                delayed(memo_oscillogram)(df, self.vendor, self.correct_baseline) for df in self.rawdata[i:hb]])
            pulsesdata = [func(df[1], *args) for df in self.rawdata[i:hb]]
            discretedata += pulsesdata
            
            print('Files ##%d-%d time %.2f s' % (i, hb, time.time() - t), end='\t')

            self.clear_rawdata(i, hb)            
            del pulsesdata
            gc.collect()

        self.hist, self.bins = np.histogram(discretedata, bins=self.histbins)
