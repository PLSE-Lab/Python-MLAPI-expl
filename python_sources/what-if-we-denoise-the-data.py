#!/usr/bin/env python
# coding: utf-8

#  3 questions:
# 
# # What if treat it as sound?
# 
# # What if we denoise the signal?
# 
# # Can we speed up the computation (even more)
# 
# 
# ## Aditional : automated stacking with vecstack

# First question:
# 
# Its time series, why not treat it as sound and do feature engineering of audio data.
# Very nice paper if you want a comprehensive survey https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S2048770314000122
# 
# 
# 
# Features I used:
# stft,mfccs,chroma,mel,tonnetz. What are they?
# 
# 1. stft- Short-time Fourier transform. -Fourier-related transform used to determine the sinusoidal frequency and phase content of local sections of a signal as it changes over time
# 
# 2. MFCCS- feature in speech recognition https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
# 
# 3. chroma https://en.wikipedia.org/wiki/Chroma_feature
# 
# 4. mel- https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
# 
# 5. tonnetz- https://sites.google.com/site/tonalintervalspace/
# 

# Second question from https://www.kaggle.com/jackvial/dwt-signal-denoising:
# 
# OK, so we could seperate signal from the noise and let the algos focus on the essentials without extracting the essentials trough new features
# 
# 
# BUT, the key is. Are we doing it correctly? In the sence that are we also removing the important signals from the data along the way?
# 
# 
# Well from local CV they are very close. I did not even modify (expect for sampling rate) the denoising functions. I would assume It could be improved. In the worst case extract the most useful engineered features from the DENOISED data set and add them to the new features

# In[ ]:





# Third question:
#     
# Why do we even care about speeding it up more? 
# Dataset is huge, in order to iterate fast and try new features, techniques etc... we should speed up the code.
# 
# First of all THANK YOU Ashi for the nice class in the discussions. That made me wonder what are some other techniques (parallelisation in python) that could be of help. I made a [compilation of that subject](https://www.kaggle.com/zikazika/parallelisation-in-python).
# 
# Where I investigated:
# 
# 1. multiprocessing library- Difference Pool and Process
# 2. Numba decorators
# 3. Joblib parallelisation (there is an option for threading)
# 4. Some relevant topics and questions that came along
# 

# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import librosa

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from vecstack import stacking
from lightgbm import LGBMRegressor


import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers


from scipy import stats
from sklearn.svm import NuSVR, SVR
from scipy.stats import kurtosis
from keras.wrappers.scikit_learn import KerasRegressor
from numba import jit, int32

import pyarrow.parquet as pq
import gc
import pywt
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter

from sklearn.preprocessing import MinMaxScaler


# In[ ]:





# In[ ]:


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=4000000):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )


# In[ ]:





# In[ ]:


class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = '../input/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('../input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y # its a generator
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        
        x_hp = high_pass_filter(x, low_cutoff=10000, sample_rate=4000000)
    

        x = denoise_signal(x_hp, wavelet='haar', level=1)
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id
        #audio
  
        sample_rate=4000000
        feature_dict['stft1']=np.abs(librosa.stft(x))
        feature_dict['stft']= np.mean(np.abs(librosa.stft(x)))

        feature_dict['mfccs']=np.mean(np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T,axis=0))

        feature_dict['chroma']=np.mean(np.mean((librosa.feature.chroma_stft(S=feature_dict['stft1'], sr=sample_rate).T)))

        feature_dict['mel']=np.mean(np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T))

        feature_dict['tonnetz']= np.mean(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x),sr=sample_rate).T))

        x=pd.Series(x)
        
        #some of mine some of andrews features

        def calc_change_rate(x):
            change = (np.diff(x) / x[:-1])
            change = change[np.nonzero(change)[0]]
            change = change[~np.isnan(change)]
            change = change[change != -np.inf]
            change = change[change != np.inf]
            return np.mean(change)


        def add_trend_feature(arr, abs_values=False):
            idx = np.array(range(len(arr)))
            if abs_values:
                arr = np.abs(arr)
            lr = LinearRegression()
            lr.fit(idx.reshape(-1, 1), arr)
            return lr.coef_[0]

        def classic_sta_lta(x, length_sta, length_lta):

            sta = np.cumsum(x ** 2)

            # Convert to float
            sta = np.require(sta, dtype=np.float)

            # Copy for LTA
            lta = sta.copy()

            # Compute the STA and the LTA
            sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
            sta /= length_sta
            lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
            lta /= length_lta

            # Pad zeros
            sta[:length_lta - 1] = 0

            # Avoid division by zero by setting zero values to tiny float
            dtiny = np.finfo(0.0).tiny
            idx = lta < dtiny
            lta[idx] = dtiny

            return sta / lta
        

        feature_dict['mean'] = x.mean()
        feature_dict['std'] = x.std()
        feature_dict['max'] = x.max()
        feature_dict['min'] = x.min()

        feature_dict['mean_change_abs'] = np.mean(np.diff(x))
        feature_dict['mean_change_rate'] = calc_change_rate(x)
        feature_dict['abs_max'] = np.abs(x).max()
        feature_dict['abs_min'] = np.abs(x).min()

        feature_dict['std_first_50000'] = x[:50000].std()
        feature_dict['std_last_50000'] = x[100000:].std()
        feature_dict['std_first_10000'] = x[:10000].std()


        feature_dict['avg_first_50000'] = x[:50000].mean()

        feature_dict['avg_first_10000'] = x[:10000].mean()


        feature_dict['min_first_50000'] = x[:50000].min()

        feature_dict['min_first_10000'] = x[:10000].min()
        feature_dict['min_last_10000'] = x[-10000:].min()

        feature_dict['max_first_50000'] = x[:50000].max()

        feature_dict['max_first_10000'] = x[:10000].max()
        feature_dict['max_last_10000'] = x[-10000:].max()

        feature_dict['max_to_min'] = x.max() / np.abs(x.min())
        feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_dict['count_big'] = len(x[np.abs(x) > 500])
        feature_dict['sum'] = x.sum()

        feature_dict['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

        feature_dict['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
        feature_dict['mean_change_rate_last_10000'] = calc_change_rate(x[:-10000])

        feature_dict['q95'] = np.quantile(x, 0.95)
        feature_dict['q99'] = np.quantile(x, 0.99)
        feature_dict['q05'] = np.quantile(x, 0.05)
        feature_dict['q01'] = np.quantile(x, 0.01)

        feature_dict['abs_q95'] = np.quantile(np.abs(x), 0.95)
        feature_dict['abs_q99'] = np.quantile(np.abs(x), 0.99)
        feature_dict['abs_q05'] = np.quantile(np.abs(x), 0.05)
        feature_dict['abs_q01'] = np.quantile(np.abs(x), 0.01)

        feature_dict['trend'] = add_trend_feature(x)
        feature_dict['abs_trend'] = add_trend_feature(x, abs_values=True)
        feature_dict['abs_mean'] = np.abs(x).mean()
        feature_dict['abs_std'] = np.abs(x).std()

        feature_dict['kurt'] = x.kurtosis()
        feature_dict['skew'] = x.skew()
        feature_dict['med'] = x.median()

        feature_dict['Hilbert_mean'] = np.abs(hilbert(x)).mean()
        feature_dict['Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
        feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        feature_dict['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        feature_dict['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        feature_dict['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        feature_dict['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        feature_dict['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        feature_dict['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
        feature_dict['Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
        ewma = pd.Series.ewm
        feature_dict['exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
        feature_dict['exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
        feature_dict['exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
        no_of_std = 3
        feature_dict['MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
        feature_dict['MA_700MA_BB_high_mean'] = (feature_dict['Moving_average_700_mean'] + no_of_std * feature_dict['Moving_average_700_mean'])
        feature_dict['MA_700MA_BB_low_mean'] = (feature_dict['Moving_average_700_mean'] - no_of_std * feature_dict['Moving_average_700_mean'])
        feature_dict['MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
        feature_dict['MA_400MA_BB_high_mean'] = (feature_dict['MA_400MA_std_mean'] + no_of_std * feature_dict['MA_400MA_std_mean'])
        feature_dict['MA_400MA_BB_low_mean'] = (feature_dict['MA_400MA_std_mean'] - no_of_std * feature_dict['MA_400MA_std_mean'])
        feature_dict['MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()


        feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict['q999'] = np.quantile(x,0.999)
        feature_dict['q001'] = np.quantile(x,0.001)
        feature_dict['ave10'] = stats.trim_mean(x, 0.1)








        
        return feature_dict

    def generate(self):
        feature_list = []
        res =Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)  for s, x, y in tqdm(self.read_chunks(), total=self.total_data)) # its a generator now, we can loop over it
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)


# In[ ]:





# Lets compare it now with a one without DWT denoising---https://www.kaggle.com/jackvial/dwt-signal-denoising

# In[ ]:


class FeatureGenerator1(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = '../input/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('../input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y # its a generator
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        
        
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id
        #audio
  
        sample_rate=4000000
        feature_dict['stft1']=np.abs(librosa.stft(x))
        feature_dict['stft']= np.mean(np.abs(librosa.stft(x)))

        feature_dict['mfccs']=np.mean(np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T,axis=0))

        feature_dict['chroma']=np.mean(np.mean((librosa.feature.chroma_stft(S=feature_dict['stft1'], sr=sample_rate).T)))

        feature_dict['mel']=np.mean(np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T))

        feature_dict['tonnetz']= np.mean(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x),sr=sample_rate).T))

        x=pd.Series(x)
        #some of mine some of andrews features

        def calc_change_rate(x):
            change = (np.diff(x) / x[:-1])
            change = change[np.nonzero(change)[0]]
            change = change[~np.isnan(change)]
            change = change[change != -np.inf]
            change = change[change != np.inf]
            return np.mean(change)


        def add_trend_feature(arr, abs_values=False):
            idx = np.array(range(len(arr)))
            if abs_values:
                arr = np.abs(arr)
            lr = LinearRegression()
            lr.fit(idx.reshape(-1, 1), arr)
            return lr.coef_[0]

        def classic_sta_lta(x, length_sta, length_lta):

            sta = np.cumsum(x ** 2)

            # Convert to float
            sta = np.require(sta, dtype=np.float)

            # Copy for LTA
            lta = sta.copy()

            # Compute the STA and the LTA
            sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
            sta /= length_sta
            lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
            lta /= length_lta

            # Pad zeros
            sta[:length_lta - 1] = 0

            # Avoid division by zero by setting zero values to tiny float
            dtiny = np.finfo(0.0).tiny
            idx = lta < dtiny
            lta[idx] = dtiny

            return sta / lta
        

        feature_dict['mean'] = x.mean()
        feature_dict['std'] = x.std()
        feature_dict['max'] = x.max()
        feature_dict['min'] = x.min()

        feature_dict['mean_change_abs'] = np.mean(np.diff(x))
        feature_dict['mean_change_rate'] = calc_change_rate(x)
        feature_dict['abs_max'] = np.abs(x).max()
        feature_dict['abs_min'] = np.abs(x).min()

        feature_dict['std_first_50000'] = x[:50000].std()
        feature_dict['std_last_50000'] = x[100000:].std()
        feature_dict['std_first_10000'] = x[:10000].std()


        feature_dict['avg_first_50000'] = x[:50000].mean()

        feature_dict['avg_first_10000'] = x[:10000].mean()


        feature_dict['min_first_50000'] = x[:50000].min()

        feature_dict['min_first_10000'] = x[:10000].min()
        feature_dict['min_last_10000'] = x[-10000:].min()

        feature_dict['max_first_50000'] = x[:50000].max()

        feature_dict['max_first_10000'] = x[:10000].max()
        feature_dict['max_last_10000'] = x[-10000:].max()

        feature_dict['max_to_min'] = x.max() / np.abs(x.min())
        feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_dict['count_big'] = len(x[np.abs(x) > 500])
        feature_dict['sum'] = x.sum()

        feature_dict['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

        feature_dict['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
        feature_dict['mean_change_rate_last_10000'] = calc_change_rate(x[:-10000])

        feature_dict['q95'] = np.quantile(x, 0.95)
        feature_dict['q99'] = np.quantile(x, 0.99)
        feature_dict['q05'] = np.quantile(x, 0.05)
        feature_dict['q01'] = np.quantile(x, 0.01)

        feature_dict['abs_q95'] = np.quantile(np.abs(x), 0.95)
        feature_dict['abs_q99'] = np.quantile(np.abs(x), 0.99)
        feature_dict['abs_q05'] = np.quantile(np.abs(x), 0.05)
        feature_dict['abs_q01'] = np.quantile(np.abs(x), 0.01)

        feature_dict['trend'] = add_trend_feature(x)
        feature_dict['abs_trend'] = add_trend_feature(x, abs_values=True)
        feature_dict['abs_mean'] = np.abs(x).mean()
        feature_dict['abs_std'] = np.abs(x).std()

        feature_dict['kurt'] = x.kurtosis()
        feature_dict['skew'] = x.skew()
        feature_dict['med'] = x.median()

        feature_dict['Hilbert_mean'] = np.abs(hilbert(x)).mean()
        feature_dict['Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
        feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        feature_dict['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        feature_dict['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        feature_dict['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        feature_dict['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        feature_dict['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        feature_dict['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
        feature_dict['Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
        ewma = pd.Series.ewm
        feature_dict['exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
        feature_dict['exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
        feature_dict['exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
        no_of_std = 3
        feature_dict['MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
        feature_dict['MA_700MA_BB_high_mean'] = (feature_dict['Moving_average_700_mean'] + no_of_std * feature_dict['Moving_average_700_mean'])
        feature_dict['MA_700MA_BB_low_mean'] = (feature_dict['Moving_average_700_mean'] - no_of_std * feature_dict['Moving_average_700_mean'])
        feature_dict['MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
        feature_dict['MA_400MA_BB_high_mean'] = (feature_dict['MA_400MA_std_mean'] + no_of_std * feature_dict['MA_400MA_std_mean'])
        feature_dict['MA_400MA_BB_low_mean'] = (feature_dict['MA_400MA_std_mean'] - no_of_std * feature_dict['MA_400MA_std_mean'])
        feature_dict['MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()


        feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict['q999'] = np.quantile(x,0.999)
        feature_dict['q001'] = np.quantile(x,0.001)
        feature_dict['ave10'] = stats.trim_mean(x, 0.1)








        
        return feature_dict

    def generate(self):
        feature_list = []
        res =Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)  for s, x, y in tqdm(self.read_chunks(), total=self.total_data)) # its a generator now, we can loop over it
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)


# In[ ]:


training_data


# In[ ]:


get_ipython().run_line_magic('time', '')

training_fg = FeatureGenerator(dtype='train', n_jobs=4, chunk_size=150000)
training_data = training_fg.generate()


# In[ ]:


y_train=training_data.target
training_data.drop(["stft1","target","seg_id"],inplace=True,axis=1)


# In[ ]:


scaler = MinMaxScaler()
training_data = pd.DataFrame(scaler.fit_transform(training_data), columns=training_data.columns)


# Dataset without DWT

# In[ ]:


get_ipython().run_line_magic('time', '')

training_fg_noDWT = FeatureGenerator1(dtype='train', n_jobs=4, chunk_size=150000)
training_data_noDWT = training_fg_noDWT.generate()
y_train_noDWT=training_data_noDWT.target
training_data_noDWT.drop(["stft1","target","seg_id"],inplace=True,axis=1)
scaler = MinMaxScaler()
training_data_noDWT = pd.DataFrame(scaler.fit_transform(training_data_noDWT), columns=training_data_noDWT.columns)


# Set data without DWT

# In[ ]:


get_ipython().run_line_magic('time', '')
test_fg_noDWT = FeatureGenerator1(dtype='test', n_jobs=4, chunk_size=None)
test_data_noDWT = test_fg_noDWT.generate()
test_data_noDWT.drop(["stft1","target","seg_id"],inplace=True,axis=1)
scaler = MinMaxScaler()
test_data_noDWT = pd.DataFrame(scaler.fit_transform(test_data_noDWT), columns=test_data_noDWT.columns)


# denoised test data:

# In[ ]:


get_ipython().run_line_magic('time', '')
test_fg = FeatureGenerator(dtype='test', n_jobs=4, chunk_size=None)
test_data = test_fg.generate()
test_data.drop(["stft1","target","seg_id"],inplace=True,axis=1)
scaler = MinMaxScaler()
test_data = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)


# There is 1 missing value:

# In[ ]:


training_data.fillna(training_data.mean_change_rate_first_10000.mean(), inplace=True)
training_data_noDWT.fillna(training_data_noDWT.mean_change_rate_first_10000.mean(), inplace=True)


# In[ ]:


test_data.fillna(test_data.mean_change_rate_first_10000.mean(), inplace=True)
test_data_noDWT.fillna(training_data_noDWT.mean_change_rate_first_10000.mean(), inplace=True)


# Vecstack, just playing around with it. No special hyperparameters or nothing

# In[ ]:


models = [
    KNeighborsRegressor(n_neighbors=5,
                        n_jobs=-1),
        
    RandomForestRegressor(random_state=0, n_jobs=-1, 
                           n_estimators=100, max_depth=3),
        
    XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1, 
                  n_estimators=100, max_depth=3),
    LinearRegression(),
    CatBoostRegressor(random_state=0),
    NuSVR(gamma='scale', nu=0.9, C=10.0, tol=0.01)
    
   
]


# In[ ]:


training_data.std_last_50000[4194]=4
training_data_noDWT.std_last_50000[4194]=4


# In[ ]:


S_train, S_test = stacking(models,                   
                           training_data, y_train, test_data,   
                           regression=True, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=mean_absolute_error, 
    
                           n_folds=10, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[ ]:


params = {'num_leaves': 128,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501
         }


# CV with stacking, 3*5 CV
# 
# Now I do realise that CV will be inflated because we used the same y_train data for the weak learners but one can use it as a benchmark for trying other things, not necessarily to correlate it to LB

# In[ ]:


skf = KFold(n_splits=5, shuffle=True, random_state=123)
oof = pd.DataFrame(y_train)
oof['predict'] = 0

val_mae = []


# In[ ]:


S_train=pd.DataFrame(S_train)
y_train1=pd.Series(y_train)


# In[ ]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(S_train, y_train1)):
    X_train, y_train = S_train.iloc[trn_idx], y_train1.iloc[trn_idx]
    X_valid, y_valid = S_train.iloc[val_idx], y_train1.iloc[val_idx]
    
    N = 3
    p_valid,yp = 0,0
    for i in range(N):
    
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(params,trn_data,1000,valid_sets = [trn_data, val_data],early_stopping_rounds=100,verbose_eval = 50,evals_result=evals_result)
        p_valid += lgb_clf.predict(X_valid)

    
    oof['predict'][val_idx] = p_valid/N
    mae = mean_absolute_error(y_valid, p_valid)
    val_mae.append(mae)

   
    
    


# In[ ]:


mae1 = mean_absolute_error(oof['target'], oof['predict'])
print("local mae1 = {}".format(mae1))
      


# Ok lets try without denoising of data.

# In[ ]:


S_train_noDWT, S_test_noDWT = stacking(models,                   
                           training_data_noDWT, y_train_noDWT, test_data_noDWT,   
                           regression=True, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=mean_absolute_error, 
    
                           n_folds=10, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[ ]:



oof_noDWT = pd.DataFrame(y_train_noDWT)
oof_noDWT['predict'] = 0

val_mae_noDWT = []


# In[ ]:


S_train_noDWT=pd.DataFrame(S_train_noDWT)
y_train_noDWT=pd.Series(y_train_noDWT)


# In[1]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(S_train_noDWT, y_train_noDWT)):
    X_train1, y_train1 = S_train_noDWT.iloc[trn_idx], y_train_noDWT.iloc[trn_idx]
    X_valid1, y_valid1 = S_train_noDWT.iloc[val_idx], y_train_noDWT.iloc[val_idx]
    
    N = 3
    p_valid,yp = 0,0
    for i in range(N):
    
        trn_data1 = lgb.Dataset(X_train1, label=y_train1)
        val_data1 = lgb.Dataset(X_valid1, label=y_valid1)
        evals_result1 = {}
        lgb_clf_noDWT = lgb.train(params,trn_data1,1000,valid_sets = [trn_data1, val_data1],early_stopping_rounds=100,verbose_eval = 50,evals_result=evals_result1)
        p_valid += lgb_clf_noDWT.predict(X_valid1)

    
    oof_noDWT['predict'][val_idx] = p_valid/N
    mae1 = mean_absolute_error(y_valid1, p_valid)
    val_mae_noDWT.append(mae1)

   
    
    


# In[ ]:


mae2 = mean_absolute_error(oof_noDWT['target'], oof_noDWT['predict'])
print("local mae1 = {}".format(mae2))
      

