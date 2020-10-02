#!/usr/bin/env python
# coding: utf-8

# ![](https://pp.userapi.com/c848636/v848636381/10387a/82TkN23uVpQ.jpg)

# # Intro
# This kernel is dedicated to exploration of LANL Earthquake Prediction Challenge. 
# I suggest trying out different methods and functions that are used to process signals for feature extraction:
# * [Hilbert transform](http:/en.wikipedia.org/wiki/Analytic_signal) 
# * Smooth a pulse using a [Hann](http://en.wikipedia.org/wiki/Hann_function) window 
# * Use trigger [classic STA/LTA](http://docs.obspy.org/tutorial/code_snippets/trigger_tutorial.html#available-methods)
# 
# Thank [Vishy](http://www.kaggle.com/viswanathravindran) for his [discuss](http://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/77267) and links.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32})")


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff


# In[ ]:


from scipy.signal import hilbert, hann, convolve


# #  [Hilbert transform](http:/en.wikipedia.org/wiki/Analytic_signal) 

#  In mathematics and signal processing, an analytic signal is a complex-valued function that has no negative frequency components.  The real and imaginary parts of an analytic signal are real-valued functions related to each other by the Hilbert transform.
# 
# The basic idea is that the negative frequency components of the Fourier transform (or spectrum) of a real-valued function are superfluous, due to the Hermitian symmetry of such a spectrum. These negative frequency components can be discarded with no loss of information, provided one is willing to deal with a complex-valued function instead.
# 
# **If you don't want to understand math, just look at how the signal function changes :)**

# In[ ]:


per = 5e-6


# In[ ]:


#Calculate Hilbert transform
signal = train.acoustic_data[:int(len(train)*per)]
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)


# In[ ]:


trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)

trace1 = go.Scatter(
    y = amplitude_envelope,
    name = 'amplitude_envelope'
)


# In[ ]:


data = [trace0, trace1]
layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")


# # Smooth a pulse using a [Hann](http://en.wikipedia.org/wiki/Hann_function) window 

# The Hann function is typically used as a window function in digital signal processing to select a subset of a series of samples in order to perform a Fourier transform or other calculations.

# In[ ]:


#Calculate Hann func
win = hann(50)
filtered = convolve(signal, win, mode='same') / sum(win)


# In[ ]:


trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)
trace3 = go.Scatter(
    y = filtered,
    name= 'filtered'
) 


data = [trace0, trace3]

layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")


#  # [classic STA/LTA](http://docs.obspy.org/tutorial/code_snippets/trigger_tutorial.html#available-methods)

# The algorithm for calculating the entry times of seismic and acoustic waves is based on the calculation of the envelope signal using the algorithm STA/LTA, widely used in the world to detect seismic signals of pulse shape.
# 
# The STA/LTA algorithm processes an already filtered signal using two time-moving Windows ("moving average"): a short time Average window ("Short time Average window") and a long time Average window ("long time Average"). STA calculates an estimate of the value of the "instantaneous" amplitude of the useful signal (for example, the p-wave of an industrial explosion), LTA estimates the average value of the noise in the long section. Next, the STA/LTA ratio of the two values calculated for each new reference signal received at the input of the algorithm is calculated.

# In[ ]:


def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


# In[ ]:


#Calculate STA/LTA
sta_lta = classic_sta_lta_py(signal, 50, 1000)


# In[ ]:


trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)


trace4 = go.Scatter(
    y = sta_lta,
    name= 'sta_lta'
) 


data = [trace0,trace4]

layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")


# Thanks for reading! If you have interesting ideas on how to work with signal data, we will discuss in the comments.
# Further ideas is the search of different frequency response characteristics and configuration of ML algorithms.
# 
# Let's solve this problem and save many lives. Good luck.
