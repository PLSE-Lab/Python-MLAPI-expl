#!/usr/bin/env python
# coding: utf-8

# # A signal processing approach - low pass filtering
# 
# Due to the nature of the signal data we have been provided, a signal processing approach can provide advantages. In short, it can help us decouple the signal from the noise in order to provide our models with a less noisy signal. In this kernel I will be performing the following on each batch:
# 1. Analyze frequency domain characteristics
# 2. Characterize Signal-To-Noise Ratio (SNR) of the data
# 3. Based on SNR, apply low pass filter to reduce signal noise
# 4. Perform feature engineering and apply models based on filtered signal
# 
# #### I am using the data without drift dataset for this kernel: https://www.kaggle.com/cdeotte/data-without-drift
# 
# ### If you find this kernel useful, please upvote!
# 

# In[ ]:


import datetime
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt,freqz
from sklearn import *
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool,CatBoostRegressor
import time
import datetime
from sklearn.model_selection import KFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('../input/data-without-drift/train_clean.csv')
test = pd.read_csv('../input/data-without-drift/test_clean.csv')
train.head()


# In[ ]:


batch_size = 500000
num_batches = 10
res = 1000 # Resolution of signal plots

fs = 10000       # sample rate, 10kHz
nyq = 0.5 * fs  # Nyquist Frequency
cutoff_freq_sweep = range(250,4750,50) # Sweeping from 250 to 4750 Hz for SNR measurement
lpf_cutoff = 600


# ### This is a good visualization of the training data signal and open channels.

# In[ ]:


plt.figure(figsize=(20,5));
plt.plot(range(0,train.shape[0],res),train.signal[0::res])
for i in range(num_batches+1): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')
for j in range(num_batches): plt.text(j*batch_size+200000,num_batches,str(j+1),size=20)
plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 
plt.title('Training Data Signal - 10 batches',size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(20,5));
plt.plot(range(0,train.shape[0],res),train.open_channels[0::res])
for i in range(num_batches+1): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')
for j in range(num_batches): plt.text(j*batch_size+200000,num_batches,str(j+1),size=20)
plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 
plt.title('Training Data Open Channels - 10 batches',size=20)
plt.show()


# ### The following functions can be used to create a high-pass, low-pass, or band-pass butterworth filter as well as measure SNR. I only used the low-pass filter in this kernel, but highly encourage others to experiment with different filter configurations.

# In[ ]:


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


# # Frequency Domain Analysis
# 
# ### First off, we want to better understand the frequency content within each batch. We notice that there is considerable broadband noise and it isn't clear from this that there is a predominant frequency that differentiates the signal. There are some batches that have some interesting features at the low frequencies (Batch 1 and 2). There also appears to be some interesting noise scatter amonst higher frequencies in batch 8. Based on this, we may be able to attenuate a large portion of the higher frequencies to reduce signal noise. We will do this by applying a low-pass filter.

# In[ ]:


fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25, 15))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for batch in range(num_batches):
    fft = sp.fftpack.fft(train.signal[batch_size*(batch):batch_size*(batch+1)])
    psd = np.abs(fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)
    i = fftfreq > 0
    
    c = next(colors)["color"]
    ax[batch].plot(fftfreq[i], 10 * np.log10(psd[i]),color=c)
    ax[batch].set_title(f'Batch {batch+1}')
    ax[batch].set_xlabel('Frequency (Hz)')
    ax[batch].set_ylabel('PSD (dB)')


# # Signal-to-Noise (SNR) Characterization 
# 
# ### We can apply low-pass filters with different cutoff frequencies in order to see at which cutoff frequency our signal begins to degrade. It looks like a cutoff frequency of 600Hz provides us with the best SNR for most batches. One thing to note is that the SNR for different batches is either positive or negative. Typically, a negative SNR means our noise is greater than our signal. However, I will show later that this isn't actual the case with some of our batches. 
# 
# **Positive SNR:** Batch 4, Batch 5, Batch 6, Batch 9, Batch 10
# 
# **Negative SNR:** Batch 1, Batch 2, Batch 3, Batch 7, Batch 8

# In[ ]:


plt.figure(figsize=(15,15));

# Filter requirements.
order = 20  
SNR = np.zeros(len(cutoff_freq_sweep))

for batch in range(num_batches):
    for index,cut in enumerate(cutoff_freq_sweep): 
        signal_lpf = butter_lowpass_filter(train.signal[batch_size*(batch):batch_size*(batch+1)], cut, fs, order)
        SNR[index] = signaltonoise(signal_lpf)
    
    plt.plot(cutoff_freq_sweep,SNR)

plt.title('Signal-to-Noise Ratio Per Batch')    
plt.xlabel('Frequency')
plt.ylabel('SNR')
plt.legend(['Batch 1','Batch 2','Batch 3','Batch 4','Batch 5','Batch 6','Batch 7','Batch 8','Batch 9','Batch 10',])


# # Low Pass Filtering By Batch
# 
# ### In case you are unfamiliar with filtering, I am plotting the the low pass filter design with a cutoff frequency at 600Hz below. Notice how we are not attenuating between 0 and 600Hz, i.e. we are allowing low frequencies to 'pass' unaffected by the filter. But anything above 600Hz will be attenuated as you can see by how the amplitude falls away all the way out to our Nyquist frequency.

# In[ ]:


b, a = butter(order, lpf_cutoff/nyq, btype='low', analog=False)
w,h = freqz(b,a, fs=fs)

plt.figure(figsize=(16,8));
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [Hz]')
plt.title('Low-pass Butterworth Filter, cutoff @ 600Hz')


# ### When we apply the filter to Batch 1, we see how our frequency response changes.

# In[ ]:


fft = sp.fftpack.fft(train.signal[batch_size*(batch-1):batch_size*batch])
psd = np.abs(fft) ** 2
fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)
i = fftfreq > 0

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(hspace = .5)
ax[0].plot(fftfreq[i], 10 * np.log10(psd[i]))
ax[0].set_xlabel('Frequency (1/10000 seconds)')
ax[0].set_ylabel('PSD (dB)')
ax[0].set_title('Unfiltered')

batch = 8
signal_lpf_batch_8 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fft = sp.fftpack.fft(signal_lpf_batch_8)
psd = np.abs(fft) ** 2
fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)
i = fftfreq > 0

ax[1].plot(fftfreq[i], 10 * np.log10(psd[i]))
ax[1].set_xlabel('Frequency (1/10000 seconds)')
ax[1].set_ylabel('PSD (dB)')
ax[1].set_title('Low pass filter - cutoff = 600 Hz')


# ### Now we will apply this Low Pass Filter (LPF) to all batches and look at how that impacts the time domain signal.

# ## Batch 1
# 
# ### We can see that our noise floor has been reduced significantly. We also see that some of the spikes have been reduced in amplitude a bit, but overall not nearly as much as our noise floor.

# In[ ]:


batch = 1

signal_lpf_batch_1 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_1[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 2
# 
# ### Same story as Batch 1, more signal and less noise

# In[ ]:


batch = 2

signal_lpf_batch_2 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_2[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 3
# 
# ### I think the LPF actually really helped with normalizing levels. Notice how now open_channel = 1 and it's associated levels in the filtered signal jump around much less than the unfiltered signal.

# In[ ]:


batch = 3

signal_lpf_batch_3 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_3[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 4
# 
# ### It isn't as clear whether the LPF helped out for this batch, but we will see when we check model performance

# In[ ]:


batch = 4

signal_lpf_batch_4 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_4[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 5
# 
# ### We may have lost some signal when filtering on this batch. 

# In[ ]:


batch = 5

signal_lpf_batch_5 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_5[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 6
# 
# ### Same as Batch 5, we may have lost some signal in the filtering

# In[ ]:


batch = 6

signal_lpf_batch_6 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_6[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 7
# 
# ### This looks a lot like Batch 3 where it is clear that filtering helped reduce noise.

# In[ ]:


batch = 7

signal_lpf_batch_7 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_7[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 8
# 
# ### Batch 8 is unique in the sense that it looks like there was some measurement noise. The LPF was able to filter it out which will help us out a lot.

# In[ ]:


batch = 8

signal_lpf_batch_8 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_8[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 9
# 
# ### Unclear whether filtering helped or hurt

# In[ ]:


batch = 9

signal_lpf_batch_9 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_9[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Batch 10
# 
# ### Unclear whether filtering helped or hurt

# In[ ]:


batch = 10

signal_lpf_batch_10 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')
ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])
ax[1].plot(range(0,batch_size,res),signal_lpf_batch_10[::res])

ax[0].legend(['open_channels'])
ax[1].legend(['signal', 'filtered signal'])


# ## Now just apply LPF to train and test data. We will start by applying our filter to one batch at a time, however we could see improvement by only applying the filter to batches we know were helped.

# In[ ]:


# Preprocess train data
batch = 8
train['signal'][batch_size*(batch-1):batch_size*batch] = signal_lpf_batch_8

# Train Data
train['signal_undrifted'] = train['signal']
# Test Data
test['signal_undrifted'] = test['signal']


# # Feature Engineering and Modeling

# In[ ]:


def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 50_000
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal_undrifted'].mean()
        d['median'+c] = df.groupby([c])['signal_undrifted'].median()
        d['max'+c] = df.groupby([c])['signal_undrifted'].max()
        d['min'+c] = df.groupby([c])['signal_undrifted'].min()
        d['std'+c] = df.groupby([c])['signal_undrifted'].std()
        d['mean_abs_chg'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())
        df['range'+c] = df['max'+c] - df['min'+c]
        df['maxtomin'+c] = df['max'+c] / df['min'+c]
        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2
    
    #add shifts
    df['signal_shift_+1'] = [0,] + list(df['signal_undrifted'].values[:-1])
    df['signal_shift_-1'] = list(df['signal_undrifted'].values[1:]) + [0]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+1'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-1'][i] = np.nan

    # add shifts_2
    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal_undrifted'].values[:-2])
    df['signal_shift_-2'] = list(df['signal_undrifted'].values[2:]) + [0] + [1]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+2'][i] = np.nan
    for i in df[df['batch_index']==1].index:
        df['signal_shift_+2'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-2'][i] = np.nan
    for i in df[df['batch_index']==49998].index:
        df['signal_shift_-2'][i] = np.nan 
        
    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal_undrifted', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal_undrifted']
        
    return df

train = features(train)
test = features(test)


# In[ ]:


def f1_score_calc(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average="macro")
    return ('KaggleMetric', score, True)


def train_model_classification(X, X_test, y, params, model_type='lgb', eval_metric='f1score',
                               columns=None, plot_feature_importance=False, model=None,
                               verbose=50, early_stopping_rounds=200, n_estimators=2000):

    columns = X.columns if columns == None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {
                    'f1score': {'lgb_metric_name': lgb_Metric,}
                   }
    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X) )
    
    # averaged predictions on train data
    prediction = np.zeros((len(X_test)))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    '''for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]'''
            
    if True:        
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)    
            
        if model_type == 'lgb':
            #model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
            #model.fit(X_train, y_train, 
            #        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
            #       verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            model = lgb.train(params, lgb.Dataset(X_train, y_train),
                              n_estimators,  lgb.Dataset(X_valid, y_valid),
                              verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, feval=lgb_Metric)
            
            
            preds = model.predict(X, num_iteration=model.best_iteration) #model.predict(X_valid) 

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
        if model_type == 'xgb':
            train_set = xgb.DMatrix(X_train, y_train)
            val_set = xgb.DMatrix(X_valid, y_valid)
            model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                                     verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)
            
            preds = model.predict(xgb.DMatrix(X)) 

            y_pred = model.predict(xgb.DMatrix(X_test))
            

        if model_type == 'cat':
            # Initialize CatBoostRegressor
            model = CatBoostRegressor(params)
            # Fit model
            model.fit(X_train, y_train)
            # Get predictions
            y_pred_valid = np.round(np.clip(preds, 0, 10)).astype(int)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

 
        oof = preds
        
        scores.append(f1_score_calc(y, np.round(np.clip(preds,0,10)).astype(int) ) )

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    #prediction /= folds.n_splits
    
    print('FINAL score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    result_dict['model'] = model
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


# In[ ]:


good_columns = [c for c in train.columns if c not in ['time', 'signal','open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

X = train[good_columns].copy()
y = train['open_channels']
X_test = test[good_columns].copy()

del train, test


# In[ ]:


params_xgb = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
          'eval_metric':'rmse'}

result_dict_xgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_xgb, model_type='xgb', eval_metric='f1score', plot_feature_importance=False,
                                                      verbose=50, early_stopping_rounds=250)


# In[ ]:


params_lgb = {'learning_rate': 0.1, 'max_depth': 7, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1}

result_dict_lgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_lgb, model_type='lgb', eval_metric='f1score', plot_feature_importance=False,
                                                      verbose=50, early_stopping_rounds=250, n_estimators=3000)


# ## Predictions

# In[ ]:


preds_ensemble = 0.50 * result_dict_lgb['prediction'] + 0.50 * result_dict_xgb['prediction']


# In[ ]:


sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
sub['open_channels'] =  np.array(np.round(preds_ensemble,0), np.int) 

sub.to_csv('submission.csv', index=False, float_format='%.4f')
sub.head(10)

