#!/usr/bin/env python
# coding: utf-8

# # Liverpool ion switching: smoothing visualization
# 
# We are asked to predict "open_channels" from "signal" data in this compeition.
# When you look the data, you can understand the important data preprocessing is how to "remove noise" from the signal.
# 
# In this kernel, I will try **various kinds of smoothing methods and see its behavior**.

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# # Load data
# 
# I use [Liverpool ion switching feather](https://www.kaggle.com/corochann/liverpool-ion-switching-feather) dataset to load the data much faster. You can also refer the kernel [Convert to feather format for fast data loading](https://www.kaggle.com/corochann/convert-to-feather-format-for-fast-data-loading).

# In[ ]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/liverpool-ion-switching-feather')\n\ntrain = pd.read_feather(datadir/'train.feather')\ntest = pd.read_feather(datadir/'test.feather')\nsample_submission = pd.read_feather(datadir/'sample_submission.feather')")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample_submission.head()


# In[ ]:


signal_array = train['signal'].values
open_channels = train['open_channels'].values

test_signal_array = test['signal'].values


# # Signal smoothing
# 
# 

# In[ ]:


import numpy
import pywt
from scipy import signal
from scipy.ndimage import zoom
from scipy.signal import savgol_filter


# Referenced from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning', same_size=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    if same_size:
        # y = y[int(window_len/2-1):-int(window_len/2)]
        y = y[(window_len - 1) // 2:-(window_len - 1) // 2]
    return y


def smooth_wave(x, window_len=11, mode='hanning'):
    """Smooth `x` wave array.

    Args:
        x (numpy.ndarray): 1-dim array.
        window_len (int): window length for smoothing. odd value is expected.
        mode (str):
            'none' for not apply smoothing.
            'savgol' for applying savgol_filter.
            'flat', 'hanning', 'hamming', 'bartlett', 'blackman' for convolve smoothing.

    Returns:
        smoothed_x (numpy.ndarray): 1-dim array, same shape with `x`.
    """
    if mode == 'none':
        return x
    elif mode == 'savgol':
        return savgol_filter(x, window_length=window_len, polyorder=2)
    else:
        return smooth(x, window_len=window_len, window=mode, same_size=True)


# In[ ]:


def filter_wave(x, cutoff=(-1, -1), N=4, filtering='lfilter'):
    """Apply low pass/high pass/band pass filter on wave `x`

    Args:
        x (numpy.ndarray): original wave array.
        cutoff (tuple): tuple of 2 int.
            1st element is for lowest frequency to pass. -1 indicates to allow freq=0
            2nd element is for highest frequency to pass. -1 indicates to allow freq=infty
        N (int): order of filter
        filtering (str): filtering method. `lfilter` or `filtfilt` method.

    Returns:
        filtered_x (numpy.ndarray): same shape with `x`, filter applied.
    """
    assert x.ndim == 1
    output = 'sos' if filtering == 'sos' else 'ba'
    if cutoff[0] <= 0 and cutoff[1] <= 0:
        # Do not apply filter
        return x
    elif cutoff[0] <= 0 and cutoff[1] > 0:
        # Apply low pass filter
        output = signal.butter(N, Wn=cutoff[1]/len(x), btype='lowpass', output=output)
    elif cutoff[0] > 0 and cutoff[1] <= 0:
        # Apply high pass filter
        output = signal.butter(N, Wn=cutoff[0]/len(x), btype='highpass', output=output)
    else:
        # Apply band pass filter
        output = signal.butter(N, Wn=(cutoff[0]/len(x), cutoff[1]/len(x)), btype='bandpass', output=output)

    if filtering == 'lfilter':
        b, a = output
        return signal.lfilter(b, a, x)
    elif filtering == 'filtfilt':
        b, a = output
        return signal.filtfilt(b, a, x)
    elif filtering == 'sos':
        sos = output
        return signal.sosfilt(sos, x)
    else:
        raise ValueError("[ERROR] Unexpected value filtering={}".format(filtering))


# # plot all
# 
# At first, let's visualize signal for train & test data.

# In[ ]:


def plot_signal(smooth_fn=None, label='smooth signal', target_indices=None, interval=50,
                separate_indices=None):
    width = 500000
    if target_indices is None:
        target_indices = np.arange(10)

    for i in target_indices:
        s = 500000 * i
        y = signal_array[s:s+width][::interval]
        t = open_channels[s:s+width][::interval]
        plt.subplots(1, 1, figsize=(18, 5))
        plt.plot(y, label='signal', zorder=1)
        if smooth_fn is not None:
            if separate_indices is None:
                y2 = smooth_fn(signal_array[s:s+width])[::interval]
            else:
                y2 = np.concatenate([smooth_fn(signal_array[s+separate_indices[i]:s+separate_indices[i+1]]) for i in range(len(separate_indices) - 1)]) [::interval]
            plt.plot(y2, label=label, zorder=1)
        plt.scatter(x=np.arange(t.shape[0]), y=t, color='green', label='label', zorder=2)
        plt.title(f'Train batch={i}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


# In[ ]:


plot_signal()


# As we can see, baseline is moving at first steps of Train batch=1, and Train batch = 6, 7, 8, 9. However the `label` is kept similar scale. So we need to remove this "baseline", called "drift".
# 
# Next, let's see test signal.

# In[ ]:


def plot_test_signal(smooth_fn=None, label='smooth signal', target_indices=None, interval=50):
    width = 500000
    if target_indices is None:
        target_indices = np.arange(4)

    for i in target_indices:
        s = 500000 * i
        y = test_signal_array[s:s+width][::interval]
        plt.subplots(1, 1, figsize=(18, 5))
        plt.plot(y, label='signal', zorder=1)
        if smooth_fn is not None:
            y2 = smooth_fn(test_signal_array[s:s+width])[::interval]
            plt.plot(y2, label=label, zorder=1)
        plt.title(f'Test batch={i}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


# In[ ]:


plot_test_signal()


# Batch 1 and 2 has signal jump for each 1/5 steps. Test batch=2 has moving baseline.

# # plot all with smoothing
# 
# Now I will demonstrate several smoothing methods, to compare how these works to estimate the "baseline".

# ## hanning
# 
# These are smoothed line using **"hanning" window**, with changing `window_len` to 1000, 10000, 100000 respectively.
# I will only plot Training batch = 1, 7 for simplicity.

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=1000, mode='hanning'), target_indices=[1, 7])


# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10000, mode='hanning'), target_indices=[1, 7])


# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=100000, mode='hanning'), target_indices=[1, 7])


# Seems `window_len=100000` is smooth but not makes too much lag, `window_len=1000` captures local change but not smooth.
# `window_len=10000` seems to be good balance.
# 
# Let's compare different window mode, 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.

# ## flat

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10000, mode='flat'), target_indices=[1, 7])


# ## hamming

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10000, mode='hamming'), target_indices=[1, 7])


# ## bartlett

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10000, mode='bartlett'), target_indices=[1, 7])


# ## blackman

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10000, mode='blackman'), target_indices=[1, 7])


# # Savgol filter
# 
# Let's try another filtering, savgol filter.

# In[ ]:


plot_signal(smooth_fn=lambda x: smooth_wave(x, window_len=10001, mode='savgol'), target_indices=[1, 7])


# Seems it is sensitive to the signal jump of Batch=1.

# # Low-pass filter
# 
# Next, let's see low-pass filtering methods.
# 
# At first, I will try several cutoff for high frequency 100, 300, 500 with "lfilter" method.

# In[ ]:


cutoff_high = 100
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='lfilter'), target_indices=[1, 7])


# In[ ]:


cutoff_high = 300
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='lfilter'), target_indices=[1, 7])


# In[ ]:


cutoff_high = 500
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='lfilter'), target_indices=[1, 7])


# When maximum cutoff frequency is high (maximum cutoff=500 in the bottom), original signal's high frequency information is kept and it is sensitive to local change.<br/>
# When we make maximum cutoff frequency to low (maximum cutoff=100 in the top), original signal's high frequency information is lost and it becomes smooth line.

# Now I will try another filtering method **"filtfilt" and "sos"**.

# In[ ]:


cutoff_high = 300
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='filtfilt'), target_indices=[1, 7])


# While we can see signal *lag* in the "lfilter" and "sos" method at the batch=1 signal jump, "filtfilt" method has less lag since it sees the signal from both direction.

# In[ ]:


cutoff_high = 300
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='sos'), target_indices=[1, 7])


# Actually for Training batch=1 jump, it is more easy to manually separate indices to calculate the signal. Below is the demonstration.

# In[ ]:


cutoff_high = 50
plot_signal(smooth_fn=lambda x: filter_wave(x, cutoff=(0, cutoff_high), filtering='filtfilt'), target_indices=[1],
            separate_indices=[0, 100000, 500000])


# # Moving average
# 
# For comparison, let's see the behavior when we use moving average.

# In[ ]:


def moving_average(a, n=50000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plot_signal(smooth_fn=lambda x: moving_average(x, n=10000), target_indices=[1],)


# We can see signal jump at the batch=1. Try using window by setting center.

# In[ ]:


def pd_rolling_mean(a, n=50000):
    df = pd.DataFrame({'signal': a})
    rolling_df = df.rolling(window=n, min_periods=1, center=True)
    return rolling_df.mean().values

plot_signal(smooth_fn=lambda x: pd_rolling_mean(x, n=10000), target_indices=[1],)


# In[ ]:


plot_signal(smooth_fn=lambda x: pd_rolling_mean(x, n=10000), target_indices=[1],
            separate_indices=[0, 100000, 500000])


# # Implementation summary & references
# 
# That's all! In summary, I used follwing methods to compute smoothed signal.
# 
# 
# 1. `numpy.convolve` for 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' window.
#  - https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
# 
# 2. `scipy.signal.savgol_filter` for 'savgol' filtering.
#  - http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.savgol_filter.html
#  
# 3. `scipy.signal` for filtering 'lfilter', 'filtfilt', 'sos'.
#  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
#  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
#  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
#  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# 
# 4. `pd.rolling` for moving average computation
# 
#  - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
