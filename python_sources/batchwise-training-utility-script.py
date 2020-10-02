"""
    Credits:
    TJ Klein
    My teammates
"""
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt,freqz
from sklearn import *
from sklearn.metrics import f1_score
import lightgbm as lgb
import time
import datetime
from sklearn.model_selection import KFold
from scipy.stats import kstest as ks

batch_size = 500000
num_batches = 10
res = 1000
fs = 10000      
nyq = 0.5 * fs  
cutoff_freq_sweep = range(250,4750,50) 
lpf_cutoff = 600

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, cutoff_low, cuttoff_high, fs, order):
    normal_cutoff_low = cutoff_low / nyq
    normal_cutoff_high = cutoff_high / nyq    
    # Get the filter coefficients 
    b, a = butter(order, [normal_cutoff_low,normal_cutoff_high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def markov_p(data):
    channel_range = np.unique(data)
    channel_bins = np.append(channel_range, 11)
    data_next = np.roll(data, -1)
    matrix = []
    for i in channel_range:
        current_row = np.histogram(data_next[data == i], bins=channel_bins)[0]
        current_row = current_row / np.sum(current_row)
        matrix.append(current_row)
    return np.array(matrix)

def add_features(df, use_kstest, distribution, columns):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 50_000
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] =  df['batch'].astype(str).str.zfill(3) + '_' + df['batch_slices'].astype(str).str.zfill(3)
    for x in ['batch','batch_slices2']:
        d = pd.DataFrame()
        d[f'mean_{x}'] = df.groupby([x]).signal_undrifted.mean()
        d[f'median_{x}'] = df.groupby([x]).signal_undrifted.median()
        d[f'mean_oc_{x}'] = df.groupby([x]).open_channels.mean()
        d[f'median_oc_{x}'] = df.groupby([x]).open_channels.median()
        d[f'maximum_signal'] = df.groupby([x]).signal_undrifted.max()
        d[f'minimum_signal'] = df.groupby([x]).signal_undrifted.min()
        d[f'maximum_oc'] = df.groupby([x]).open_channels.max()
        d[f'minimum_oc'] = df.groupby([x]).open_channels.min()
        d['mean_abs_chg'+x] = df.groupby([x])['signal_undrifted'].apply(lambda c: np.mean(np.abs(np.diff(c))))
        d['abs_max'+x] = df.groupby([x])['signal_undrifted'].apply(lambda c: np.max(np.abs(c)))
        d['abs_min'+x] = df.groupby([x])['signal_undrifted'].apply(lambda c: np.min(np.abs(c)))
        d['mean_abs_chg_oc'+x] = df.groupby([x])['open_channels'].apply(lambda c: np.mean(np.abs(np.diff(c))))
        d['abs_max_oc'+x] = df.groupby([x])['open_channels'].apply(lambda c: np.max(np.abs(c)))
        d['abs_min_oc'+x] = df.groupby([x])['open_channels'].apply(lambda c: np.min(np.abs(c)))
        
        if use_kstest == True:
            if distribution == 'norm':
                if columns == 'open_channels':
                    print (f"Using KSTest with normal distribution on column open_channels on dataset {df}")
                    d[f'kstest_{x}'] = df.groupby([x])['open_channels'].apply(lambda c: ks(np.array(df.groupby([x]).open_channels), 'norm'))
                    
                if columns == 'signal_undrifted':
                    print (f"Using KSTest with normal distribution on column signal_undrifted on dataset {df}")
                    d[f'kstest_{x}'] = df.groupby([x])['signal_undrifted'].apply(lambda c: ks(np.array(df.groupby([x]).signal_undrifted), 'norm'))
                    
            elif distribution == 'uniform':
                print ("Using KSTest with uniform distribution")
                d[f'kstest_{x}'] = df.groupby([x])['signal_undrifted'].apply(lambda c: ks(np.array(df.groupby([x]).signal_undrifted), 'uniform'))
                
            elif distribution == 'norm_uniform':
                print('Both distributions are in use')
                d[f'kstest_norm_{x}'] = df.groupby([x])['signal_undrifted'].apply(lambda c: ks(np.array(df.groupby([x]).signal_undrifted), 'norm'))
                d[f'kstest_uniform_{x}'] = df.groupby([x])['signal_undrifted'].apply(lambda c: ks(np.array(df.groupby([x]).signal_undrifted), 'uniform'))
                
            else:
                print("VOID")
                pass
            
        elif use_kstest == False:
            print ("Not using KSTest")
            # d[f'kstest_{x}'] = df.groupby([x])['signal_undrifted'].apply(lambda c: ks(np.array(df.groupby([x]).signal_undrifted)), 'norm')
        else:
            print ("VOID")
        
        print(d)
        return d
        return df