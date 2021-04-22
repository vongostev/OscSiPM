![OscSiPM](https://github.com/vongostev/OscSiPM/workflows/OscSiPM/badge.svg?branch=main) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/vongostev/OscSiPM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/vongostev/OscSiPM/context:python)

# OscSiPM
Instruments to make a photocounting statistics from histograms and raw oscillograms (made by LeCroy oscilloscope or old Tektronix oscilloscope) of SiPM signal. One can correct the baseline of the oscillogram and compensate a crosstalk noise in the photocounting statistics.

# Installation
OscSiPM is available at pip. It may be installed with
```bash
pip install oscsipm
```
# How to use?
Import necessary modules:
```python
from oscsipm import PulsesHistMaker, QStatisticsMaker, find_pcrosstalk, compensate
```
Import an experimental data
```python
datadir = "C:\\expdata\\"
parser = PulsesHistMaker(datadir, vendor='tek', parallel=True, parallel_jobs=2)
parser.read()
```
Make a histogram
```python
histfile = "C:\\histograms\\test.txt"
parser.multi_pulse_histogram(frequency=1e6, time_window=10e-9)
parser.save_hist(histfile)
```
Make a photocounting statistics
```python
histfile = "C:\\histograms\\test.txt"
qmaker = QStatisticsMaker(histfile, discrete=0.021, method='fit')
Q = qmaker.getq()
```
Determine a crosstalk probability (if an optical signal is coherent) and compensate it
```python
PDE = 0.4
pcrosstalk, res = find_pcrosstalk(Q, PDE, N=50)
Q1 = compensate(Q, pcrosstalk)
```
# Requirements
See requirements.txt
