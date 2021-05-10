# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:25:54 2019

@author: Pavel Gostev
"""
from os.path import isfile, join
from os import listdir
import time
from random import sample as rsample

import gc

from dataclasses import dataclass

import lecroyparser
import tekwfm2 as tek

from scipy.sparse.linalg import spsolve
from scipy import sparse, fftpack
from scipy.signal import find_peaks
import pyfftw

import numpy as np
import numba as nb

from compress_pickle import dump, load
from joblib import Parallel, delayed
from itertools import islice


gc.enable()

VENDOR_EXT = {'tek': 'wfm',
              'lecroy': 'trc'}

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()


def sqv_bandpass_filter(data: np.ndarray, nyq: float,
                        low_cut_freq: float, high_cut_freq: float) -> np.ndarray:
    spectrum = fftpack.rfft(data)
    slen = len(spectrum)
    n_low = int(low_cut_freq / nyq * slen)
    n_high = int(high_cut_freq / nyq * slen)
    spectrum[:n_low] = 0
    spectrum[n_high:] = 0
    return np.abs(fftpack.irfft(spectrum))


def parallelize(func, data, n_jobs, *func_args):
    return Parallel(n_jobs=n_jobs)(delayed(func)(d, *func_args) for d in data)


def parse_file(datafile: str, vendor: str) -> object:
    if vendor == 'lecroy':
        data = lecroyparser.ScopeData(datafile)
    if vendor == 'tek':
        data = tek.ScopeData(datafile)
    return data


def list_files(datadir: str, vendor: str, fsoffset: int, fsnum: int) -> list:
    lf = [join(datadir, f) for f in listdir(datadir)
          if isfile(join(datadir, f)) and f.endswith(VENDOR_EXT[vendor])]
    if fsoffset != 0:
        lf = lf[fsoffset:]
    if fsnum != 0:
        lf = lf[:fsnum]
    return lf


def parse_files(oscfiles: list, vendor: str, fsnum: int = 0) -> list:
    if fsnum > 0:
        oscfiles = oscfiles[:fsnum]
    return [parse_file(df, vendor) for df in oscfiles]


def windowed(data: object, div_start: float, div_width: float) -> tuple:
    div_points = len(data.x) / data.horizInterval
    wstart = int(div_start * div_points)
    wwidth = int(div_width * div_points)
    return data.x[wstart:wstart + wwidth], data.y[wstart:wstart + wwidth]


def baseline_als(y: np.ndarray, lam: float, p: float, niter: int = 2) -> np.ndarray:
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


def correct_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.1) -> np.ndarray:
    baseline_values = baseline_als(y, lam=lam, p=p)
    return y - baseline_values


@nb.jit(nopython=True, nogil=True, fastmath=True)
def single_pulse(osc: np.ndarray, method: str = 'max') -> float:
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
def random_pulse(osc: np.ndarray, dt: float,
                 peak_height_min: float,
                 peak_width: float, peak_distance: float,
                 method: str = 'max') -> float:
    if method == 'sum':
        return np.sum(osc, dtype=float)
    elif method == 'max':
        y = osc[osc > 0]
        peaks = find_peaks(y, height=(peak_height_min, None),
                           distance=peak_distance / dt,
                           width=(peak_width / dt, None))[1]['peak_heights']
        return np.sum(peaks)


@nb.jit(nopython=True, nogil=True, fastmath=True)
def _periodic_pulse(oscarray: np.ndarray, dx: float, frequency: float,
                    time_window: float, method: str = 'max') -> list:
    points_period = int(1 / frequency / dx) + 1
    points_window = int(time_window / dx) + 1

    init_point = np.argmax(oscarray)
    pulses_points = np.append(
        np.arange(init_point, 0, -points_period)[:: -1],
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
        pulses[i] = single_pulse(oscarray[low: top], method=method)

    return pulses


def periodic_pulse(data: object, frequency: float, time_window: float,
                   method: str = 'max') -> np.ndarray:
    return _periodic_pulse(
        data.y, data.horizInterval, frequency, time_window, method)


def scope_unwindowed(data: object, time_window: float, peak_height_min: float,
                     peak_width: float, peak_distance: float, method: str = 'max') -> np.ndarray:
    points_discrete = int(time_window // data.horizInterval) + 1
    split_indexes = np.arange(0, len(data.y), points_discrete)
    chunks = np.array(np.split(data.y.copy(), split_indexes), dtype='O')
    return random_pulse(chunks, data.horizInterval, peak_height_min,
                        peak_width, peak_distance, method=method)


def from_memo(path: str, oscdata: np.ndarray, vendor: str) -> object:
    _oscdata = parse_file(path, vendor)
    _oscdata.y = oscdata
    return _oscdata


def memo_oscillogram(data: tuple, vendor: str,
                     lf_filtering: float, correct_bs: bool = True) -> tuple:
    path, oscdata = data

    if oscdata is None:
        oscdata = parse_file(path, vendor)
        y = oscdata.y
        y -= np.min(y)
        y = correct_baseline(y) if correct_bs else y
        y = sqv_bandpass_filter(y, 0.5 / oscdata.horizInterval,
                                lf_filtering, 0.5 / oscdata.horizInterval)
        oscdata.y = y
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
        Number of bins in the histogram.
        Can be change in methods 'make_hist' and 'get_hist'
    histpoints: int = 0
        Number of points are accountable in histogram
    correct_baseline: bool = True
        Correct baseline of not?
    method : str = 'max'
        Method of pulses counting. Can be 'max' or 'sum'
    vendor: str = 'lecroy'
        Vendor name. Can be 'tek' or 'lecroy'
    disp: bool = False
        Display progress of data processing
    """

    datadir: str
    fsnum: int = 0
    fsoffset: int = 0
    fchunksize: int = 10
    parallel: bool = False
    parallel_jobs: int = -1
    memo_file: str = ''
    histbins: int = 2000
    histpoints: int = 0

    correct_baseline: bool = True
    lf_filtering: float = 1e3

    method: str = 'max'
    methods: tuple = ('sum', 'max')

    vendor: str = 'lecroy'
    vendors: tuple = ('lecroy', 'tek')

    disp: bool = False

    def __post_init__(self):

        if self.vendor not in self.vendors:
            raise ValueError(
                f'{self.__name__}.vendor must be in {self.vendors}, not {self.vendor}')
        if self.method not in self.methods:
            raise ValueError(
                f'{self.__name__}.method must be in {self.methods}, not {self.method}')
        if not self.parallel:
            self.parallel_jobs = 1

        self.rawdata = {}
        self.discretedata = {}

    def read(self, fsnum: int = 0, fsoffset: int = 0,
             random_addition: bool = True, random_read: bool = True):
        """
        Load files from the given directory 'self.datadir'.

        - If 'fsnum' is 0, all files will be loaded.
        - If 'fsoffset' is 0, all files will be loaded.

        Parameters
        ----------
        fsnum : int, optional
            Number of files to load. The default is 0.
        fsoffset : int, optional
            Offset from the first file to start counting of files.
            The default is 0.
        random_addition : bool, optional
            Read random files from the self.datadir if 'fsnum' is more than files number till the end of these. 
            The default is True.
        random_read : bool, optional
            Read random files, not from the ordered list of files.
            The default is True.

        Returns
        -------
        None.

        """
        if fsnum != 0:
            self.fsnum = fsnum
        if fsoffset != 0:
            self.fsoffset = fsoffset

        datapaths = list_files(
            self.datadir, self.vendor, self.fsoffset, self.fsnum)

        if random_read or (len(datapaths) < self.fsnum and random_addition):
            _datapaths = list_files(self.datadir, self.vendor, 0, 0)
            indxs = np.arange(len(_datapaths))
            rnd_indxs = rsample(
                indxs[indxs != self.fsoffset].tolist(), self.fsnum - 1)
            datapaths = [_datapaths[i] for i in rnd_indxs + [self.fsoffset]]

        # Check if file was processed
        files_processed = [k for k, v in self.rawdata.items() if v is not None]
        files_to_remove = [k for k in self.rawdata if k not in datapaths]

        for path in files_to_remove:
            del self.rawdata[path]

        for path in datapaths:
            if path not in files_processed:
                self.rawdata[path] = None

        if self.memo_file:
            self.rawdata.update(load(self.memo_file, compression='lzma',
                                     set_default_extension=False))

        self.filesnum = len(self.rawdata) if not fsnum else fsnum

    def save_memo(self, filename: str):
        for k, d in self.rawdata.items():
            self.rawdata[k] = d.y
        dump(self.rawdata, filename, compression='lzma',
             set_default_extension=False)

    def save_hist(self, filename: str):
        np.savetxt(filename, np.vstack((self.bins[: -1], self.hist)).T)

    def single_pulse_hist(self, div_start: float = 5.9, div_width: float = 0.27):
        self.discretedata = {}
        for path, d in self.rawdata.items():
            x, y = windowed(d, div_start, div_width)
            self.discretedata.update({path: single_pulse(y, self.method)})
        self.make_hist(self.histbins)

    def multi_pulse_histogram(self, frequency: float = 2.5e6, time_window: float = 7.5e-9):
        self.parse(periodic_pulse, frequency, time_window, self.method)
        self.make_hist(self.histbins)

    def unwindowed_histogram(self, time_window: float = 100e-9,
                             peak_height_min: float = 0.01,
                             peak_width: float = 2e-9,
                             peak_distance: float = 19e-9):
        """
        Make a histogram from complete oscillogram files without windowing.
        Parameters depend from pulses' form

        Parameters
        ----------
        time_window : float, optional
            Time window size to detect pulses (in seconds). 
            The default is 100e-9.
        peak_height_min : float, optional
            Minimal pulse height to detect (in seconds). 
            The default is 0.01.
        peak_width : float, optional
            Minimal pulse width to detect (in seconds). 
            The default is 2e-9.
        peak_distance : float, optional
            Minimal distance between consecutive pulses (in seconds). 
            The default is 19e-9.

        """
        self.parse(scope_unwindowed, time_window, peak_height_min,
                   peak_width, peak_distance, self.method)
        self.make_hist(self.histbins)

    def parse(self, func, *args):
        self.discretedata = {}

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)
            def sl(it): return islice(it, i, hb)

            chunk_data = parallelize(
                memo_oscillogram, sl(self.rawdata.items()),
                self.parallel_jobs, self.vendor,
                self.lf_filtering, self.correct_baseline)
            self.rawdata.update(dict(chunk_data))

            chunk_pulses = parallelize(
                func, sl(self.rawdata.values()), self.parallel_jobs, *args)
            self.discretedata.update(
                dict(zip(sl(self.rawdata.keys()), chunk_pulses)))

            if self.disp:
                print(f'Files ##{i + 1}-{hb + 1} T={time.time() - t:.2f} s',
                      end='\t')

            del chunk_data
            del chunk_pulses
            gc.collect()

    def make_hist(self, histbins: int):
        pulses_data = np.concatenate(
            tuple(self.discretedata.values()))[:self.histpoints]
        self.hist, self.bins = np.histogram(pulses_data, bins=histbins)

    def get_hist(self, histbins: int):
        self.make_hist(histbins)
        return self.bins[:-1], self.hist
