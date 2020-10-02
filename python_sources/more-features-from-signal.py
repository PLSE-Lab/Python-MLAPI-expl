#!/usr/bin/env python
# coding: utf-8

# # This Script has few functions for generating features from the signal data we have for each signal. Most of the Help is taken from tsfresh package.

# In[ ]:


import pandas as pd, numpy as np
import scipy.stats as ss, statsmodels,pywt
from scipy import signal


# In[ ]:


param1 =[{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'},{'coeff': 2, 'attr': 'abs'},{'coeff': 3, 'attr': 'abs'},
       {'coeff': 4, 'attr': 'abs'},{'coeff': 5, 'attr': 'abs'},{'coeff': 6, 'attr': 'abs'},{'coeff': 7, 'attr': 'abs'},
       {'coeff': 8, 'attr': 'abs'},{'coeff': 9, 'attr': 'abs'},{'coeff': 10, 'attr': 'abs'},{'coeff': 11, 'attr': 'abs'},
        {'coeff': 12, 'attr': 'abs'},{'coeff': 13, 'attr': 'abs'},{'coeff': 14, 'attr': 'abs'},{'coeff': 15, 'attr': 'abs'}]
param2 = [{"aggtype": "centroid"},{"aggtype": 'variance'},{'aggtype':'skew'},{'aggtype':'kurtosis'}]


# In[ ]:


# x is the each individual signal
def width_char(x):
    indices = signal.find_peaks(x)[0]
    character = signal.peak_widths(x,indices)
    return character[0],character[1]

def prominences(x):
    indices = signal.find_peaks(x)[0]
    Prominence = signal.peak_prominences(x,indices)
    Prominence = Prominence[0]
    return Prominence.mean(),Prominence.max(),Prominence.min(),Prominence.std(),Prominence.var()

def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)

def complexity(x,normalize=True):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0
    x = np.diff(x)
    return np.sqrt(np.dot(x, x))

def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def non_linearity(x,lag):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0:(n - 2 * lag)])
    
def binned_entropy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=4)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)

def range_count(x, min, max):
    return np.sum((x >= min) & (x < max))

def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)

def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def energy_wavelet(x):
    decomposition = pywt.wavedec(x,'db4')[0]
    return decomposition,np.sqrt(np.sum(np.array(decomposition) ** 2)) / len(decomposition)

def energy_detailed_coeff(x):
    cA,cD = pywt.dwt(x,'db4')
    return np.sqrt(np.sum(np.array(cD) ** 2)) / len(cD)

# for fft_aggregated , use param2
def fft_aggregated(x, param):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert set([config["aggtype"] for config in param]) <= set(["centroid", "variance", "skew", "kurtosis"]),         'Attribute must be "centroid", "variance", "skew", "kurtosis"'


    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \sum_i[index(y_i)^moment * y_i] / \sum_i[y_i]
        
        :param y: the discrete distribution from which one wants to calculate the moment 
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y))**moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid 
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float 
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance 
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float 
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition
        
        :param y: the discrete distribution from which one wants to calculate the skew 
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float 
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 3) - 3*get_centroid(y)*variance - get_centroid(y)**3
            ) / get_variance(y)**(1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments
        
        :param y: the discrete distribution from which one wants to calculate the kurtosis 
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float 
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 4) - 4*get_centroid(y)*get_moment(y, 3)
                + 6*get_moment(y, 2)*get_centroid(y)**2 - 3*get_centroid(y)
            ) / get_variance(y)**2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis
    )

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in param]
    index = ['aggtype_"{}"'.format(config["aggtype"]) for config in param]
    return zip(index, res)

# for fft_coefficient use param1
def fft_coefficient(x, param):

    assert min([config["coeff"] for config in param]) >= 0, "Coefficients must be positive or zero."
    assert set([config["attr"] for config in param]) <= set(["imag", "real", "abs", "angle"]),         'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    res = [complex_agg(fft[config["coeff"]], config["attr"]) if config["coeff"] < len(fft)
           else np.NaN for config in param]
    index = ['coeff_{}__attr_"{}"'.format(config["coeff"], config["attr"]) for config in param]
    return zip(index, res)

