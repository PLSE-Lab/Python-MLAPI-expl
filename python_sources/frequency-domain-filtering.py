#!/usr/bin/env python
# coding: utf-8

# # Frequency Domain Noise: Causes and Remedies
# A generation ago I was an instrumentation engineer tasked with recording physical events of many types. In our contest the task is to measure ion flow across a sensor. I assume the sensor needs an amplifier since the measurements are in pico-amperes (very small current). After reviewing the test and train signals, there are several 'systemic' types of noise being added to our small signal.
# 1. Many of the signals have a **DC bias**. This is as though a fixed current is added to the signal. The practical remedy is to eliminate ground loops in cabling and in the amplifier. A DC Offset or DC bias of a signal is it's zero frequency term in the frequency domain. Since no information is contained in f=0, it can be filtered from the frequency domain to eliminate it from a signal.
# 2. The obvious **sawtooth noise** is easy to filter out in the frequency domain. A sawtooth in the frequency domain is described by the DC term f0, f1 the frequency of the tooth (1/T) and odd harmonics of f1. 
# 3. The very long, **1/2 cycle sine wave noise** is very low frequency. Half sine waves are composed of f0, and the even harmonics of the half sine frequency.
# 4. Most puzzling is the occurances of what looks like a signal beyond its normal level. It looks like the amplifier gain suddenly switches to a higher level. I do not have a remedy for this apparent issue. Practically you calibrate and zero your amplifier.
# 5. In Liverpool they use **50Hz electrical power**. Nearby the measurement is some electrical power equipment inducing current into the probe or cable. This might be an amplifier issue too. The practical remedy is to use shielded cables. To reduce this 50Hz buzz in the data, I have applied a notch filter at 50Hz. This noise is easy to see and reduce in the frequency domain of the signal.
# 6. In the higher modes of open channels, like batch 5 and 10 of the training set, it looks like the cell may be vibrating. Perhaps this vibration can be seen as the natural frequency and harmonics of the cell itself. Perhaps these modes are reached by increawsing the current or voltage of the probe. I woner if this could be approximated somehow with a hollow shpere for finding frequencies of vibration.
# 
# # Physics and Features
# Ions are charged particles. I assume that when an ion, a unit of charge, is moved across an electric potential a current is induced in the probe. Work/energy is required to move a charge in the field. I assume the work is proportional to current squared. So the total number of open channels is proportional to the signal squared. (wish I had better handle on this). Part of feature selection should involve these physics.
# 
# Much is said in discussions about Markov Chains that are involved in generating training data. I made a sketch of a Markov Chain model and it helped me think about the problem in a much differnt way: I should make models for Open Channels, y, and Open Channel Change, dy/dt.
# 
# ![Markov Model](https://www.elmtreegarden.com/wp-content/uploads/2020/04/Markov-model.jpg)
# 
# 
# My Features for modeling are:
# 1. The signal and filtered signals
# <br>a. The signal is filtered to reduce very low frequencies (VLF)
# <br>b. AND the 50Hz power noise is reduced<br>
# 2. The power of the signals: power(t) = signal(t)**2
# 3. The relative change of power of the signals is the relative work: work(t) = power(t) - power(mean)
# 4. Recent Energy is 'lev work50 100' which is Energy in the filtered signal in the past 1 millisecond
# 
# References are not completely documented yet.

# I picture the laws of motion of the Open Ion Channels, y(t), driven by the signal f(t) will take the form of a second order differencial equation. Both rolled ou and filtered signals are important in models.
# 
#        y(t) = a0
#                + a1 SUM(f(t))  - these are 'rolled-up' features (sum of current is charge q)
#                + a2 f(t)       - the signal and filtered signal (current)
#                + a3 f'(t)      - the first derivative of the signal (gradient of current)
#                
# and dy/dt is: 
# 
#        y'(t) =  a1 f(t)
#                + a2 f'(t)
#                + a3 f"(t)
# 
# and the system is constrained by the law of conservation of energy, eT + ef = 0 
# where energy of ion transmission eT = k y'(t), and
# where energy of the signal ef = R f(t)^2, so that
# 
#         R f(t)^2 - k y'(t) = 0
#         y'(t) = R/k f(t)^2

# In[ ]:


import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as sp
import scipy.fftpack
from scipy import signal
from scipy.signal import butter, sosfiltfilt, freqz, filtfilt
from sklearn import tree
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import gc
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

DATA_PATH = "../input/liverpool-ion-switching"

train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
#submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))


# In[ ]:


# Make a few new variables for modeling

def make_features(x, names, n):
    count = len(names)
    countx = len(x.columns)

    for name in names:
        
        x[name + "_dt"] = x[name] - x[name].shift(1)
        x[name + "_dt"][0] = 0.
        
        
        x[name + "_power"] = x[name]**2
        x[name + "_dt_power"] = x[name + '_dt']**2
        x[name + "_power_dt"] = x[name + "_power"] - x[name + "_power"].shift(1)
        x[name + "_power_dt"][0] = 0.
        x[name + "_rel_work"] = np.sqrt(x[name + "_power"]**2 + (x[name + "_power"].mean())**2)
        x[name + "_dt_rel_work"] = np.sqrt(x[name + "_dt_power"]**2 + (x[name + "_dt_power"].mean())**2)
        #x[name + "_rel_work2"] = x[name + "_power"] - x[name + "_power"].mean()
        
        # Discarded features
        #x[name + "_exp"] = np.exp(x[name].values)
        #x[name + "_exp"] = np.exp(x[name + "_dt"].values)
        #x[name + "_dt2"] = x[name + "_dt"] - x[name + "_dt"].shift(1)
        #x[name + "_dt2"][0] = 0.
        #x['rms_' + name] = np.sqrt(x[name]**2 + (x[name].mean())**2)
        #x['rms1_' + name] = np.sqrt(abs(x[name]**2 - (x[name].mean())**2))
        #x["rel_" + name ] = x[name] - x[name].mean()
        #x["dt_" + name + "_rel_work" ] = x[name + "_rel_work"] - x[name + "_rel_work"].shift(1)
        #x["dt_" + name + "_power" ] = x[name + "_power"] - x[name + "_power"].shift(1)
    
    
    x['50Hz_energy_floor'] = x['signal_f_50Hz_rel_work'].rolling(window=100, min_periods=5).min()
    
    x['dt_50Hz_energy_floor'] = x['signal_f_50Hz_dt_rel_work'].rolling(window=100, min_periods=5).min()
    
    x = x.drop(columns = ['signal_rel_work', 'signal_f_50Hz_rel_work','signal_dt_rel_work',
               'signal_f_50Hz_dt_rel_work','signal_dt', 'signal_f_50Hz_dt'])
    
    # Discarded features
    #x['Log_50Hz_energy_floor'] = np.log(x['50Hz_energy_floor'])
    #x['energy_floor'] = x['signal_rel_work'].rolling(window=100, min_periods=5).min()
    #x['dt_energy_floor'] = x['signal_dt_rel_work'].rolling(window=100, min_periods=5).min()
    #x['energy_rr'] = x['signal_f_50Hz_power'].rolling(window=100, min_periods=5).mean()
    
    #x['lev_50Hz'] = x['signal_f_50Hz'].rolling(window=11, min_periods=1).mean()
    #x['lev_work_100'] = x['signal_rel_work'].rolling(window=100, min_periods=1).min()
    #x['lev_work'] = x['signal_f_50Hz_rel_work'].rolling(window=100, min_periods=5).mean()
    
    # By examining lev_work50's minimum, establish the different modes of operation
    #bins = [0,0.12,0.25,0.5,1.25,1.5,2.5,100]
    #work_d = pd.cut(x['lev_work50_100'], bins=bins, labels=False)
    #x['power_mode'] = work_d
    
    x = x.replace([np.inf, -np.inf], np.nan)    
    x.fillna(0, inplace=True)
    counta = len(x.columns)
    all_features = x.columns
    new_features = all_features[countx:counta+1]
    return x, new_features


# In[ ]:


bs = 500000
fs=10000.
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25, 20))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()
print("The four batches of test signal in the Frequency Domain")
print("The powerful spikes are identifiable noise. 50Hz is Euro power frequency noise.")
print("The central spike at f=0 is DC bias and low frequency waveform noise.")
for batch in range(10):
    fft = sp.fftpack.fft(train_df.signal[bs*(batch):bs*(batch+1)])
    psd = np.abs(fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)
    
    i = abs(fftfreq) < 5000
    c = next(colors)["color"]
    ax[batch].grid()
    ax[batch].plot(fftfreq[i], np.log10(psd[i]),color=c, linewidth=.5)
    ax[batch].set_title(f'Batch {batch+1}')
    ax[batch].set_xlabel('Frequency (Hz)')
    ax[batch].set_ylabel('PSD (dB)')
    ax[batch].set_ylim(4,10)


# In[ ]:


sfft_10 = []
ws = 11
window = signal.blackmanharris(ws)
window_10 = signal.blackmanharris(ws*5)
window_small = signal.blackmanharris(5)
bs = int(bs)

# add small white noise to signal to prevent overfitting and improve results
#ref: Alexander Lakaro
STD = 0.01
old_data = train_df['signal']
new_data = old_data + np.random.normal(0,STD,size=len(train_df)) 
train_df['signal'] = new_data

for ii in range(10):  # perform filters
    i = ii*bs

    # f_dc filter: Apply Blackman-Harris high pass filter to reduce very low frequency noise for waveform noise
    # the index of 0.5Hz is 25 
    fourier = rfft(train_df.iloc[i:i+bs,1])
    for i in range(25):
        fourier[i] = fourier[i]*(window_10[i])
        fourier[i+int(bs/4)] = fourier[i+int(bs/4)]*(window_10[i])
    
    # f_50Hz filter: Apply Blackman-Harris notch filter to cut out 50Hz buzz noise and third harmonic
    # where index 2500 = 50Hz

    n = 2500
    fourier[n-28:n+27] = fourier[n-28:n+27]*(1-window_10)
    fourier[n-28+int(bs/4):n+27+int(bs/4)] = fourier[n-28+int(bs/4):n+27+int(bs/4)]*(1-window_10)
    #fourier[3*n-28:3*n+27] = fourier[3*n-28:3*n+27]*(1-window_10)
    #fourier[3*n-28+int(bs/4):3*n+27+int(bs/4)] = fourier[3*n-28+int(bs/4):3*n+27+int(bs/4)]*(1-window_10)
    
    # remove 100Hz + n * 100Hz as seen in batch 8
    # add up the 100Hz power. Just use positive frequencies for test but apply to pos and neg spectrum
    p100 = 0.
    n = 50
    for freq in range(100*n, 4900*n, 100*n):
        p100 += np.sum(abs(fourier[freq-5:freq+6] * window)**2)
    p = np.sum(abs(fourier[0:int(bs/4)])**2)
    #print('100Hz power in Batch ',ii+1, ' is ', p100, '. % of Total is ', p100/p)
    if p100/p >.001:
        print('batch ',ii+1, ' gets 100Hz filter')
        K = 5000
        sf_10 = irfft(fourier)
        sf = pd.DataFrame(sf_10, columns =['sf'])
        sf['sf'] = sf['sf'] - sf['sf'].shift(K)
        sf.loc[0:K-1,'sf'] = sf_10[0:K]
        sf_10 = sf.sf.values
        #for freq in range(100*n, 2000*n, 100*n):
            #fourier[freq-28:freq+27] = fourier[freq-28:freq+27]*(1-window_10)
            #fourier[freq-28+int(bs/4):freq+27+int(bs/4)] = fourier[freq-28+int(bs/4):freq+27+int(bs/4)]*(1-window_10)
    else:
        sf_10 = irfft(fourier)
    
    sfft_10 = np.append(sfft_10,sf_10)

train_df['signal_f_50Hz'] = 0.
train_df['signal_f_50Hz'] = sfft_10

names_in = ['signal','signal_f_50Hz']

features = pd.DataFrame()
feats = pd.DataFrame()
for ii in range(10):# Produce engineered features 
    i = ii*bs
    feats, names_out = make_features(train_df.iloc[i:i+bs,[1,3]], names_in,10)
    features = pd.concat([features,feats], axis=0)
for name in names_out:
    train_df.loc[:,name] = 0.
    train_df.loc[:,name] = features.loc[:,name].values
del features, feats

Prepare data for Modeling
# In[ ]:



Y = train_df['open_channels'].values

X = train_df.drop(['time','open_channels'], axis=1)
X = X.replace([np.inf, -np.inf], np.nan)    
X.fillna(0, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

grd = tree.DecisionTreeClassifier(max_leaf_nodes=2000)
#multi_target_tree = MultiOutputClassifier(grd, n_jobs=-1)
grd.fit(X_train, Y_train)
y_pred = grd.predict(X_test)

f1 = f1_score(Y_test,y_pred,average='macro')
print('Y (open_channels) model has f1 validation score =',f1)


# In[ ]:



# Feature strength or importance

feats = pd.DataFrame(grd.feature_importances_, columns=["importance"],index=X.columns)
feats.sort_values('importance', ascending=False)


# In[ ]:


# Fit on all batches and find most prominent errors

y_pred = grd.predict(X)
train_df['y_pred'] = y_pred
train_df['y_error'] = (train_df['open_channels'] - train_df['y_pred'])

f1 = f1_score(Y,y_pred,average='macro')
print('The f_1 score for Model of Y is ',f1)


# In[ ]:


examples = ['signal','signal_f_50Hz','y_error', 'signal_dt_power', 'signal_f_50Hz_dt_power','dt_50Hz_energy_floor']
fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 28))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(len(examples)):
    
    c = next(colors)["color"]
    ax[i].grid()
    if examples[i] in ['dt_50Hz_energy_floor','50Hz_energy_floor']:
        ax[i].plot(train_df['time'], train_df[examples[i]],color=c, linewidth= 2)
        ax[i].set_ylim(0,4)

    ax[i].plot(train_df['time'], train_df[examples[i]],color=c, linewidth=.5)
    ax[i].set_title(examples[i], fontsize=24)
    ax[i].set_xlabel('Time (seconds)', fontsize=18)
    #ax[i].set_ylabel('current (pA)', fontsize=24)
    #ax[i].set_ylim(0,5)


# In[ ]:


s , t = 3500000 , 40000000
f1_seg = f1_score(train_df.loc[s:t,'open_channels'],train_df.loc[s:t,"y_pred"],average='macro')
print(f1_seg)


# In[ ]:


sfft_10 = []
ws = 11
window = signal.blackmanharris(ws)
window_10 = signal.blackmanharris(ws*5)
window_small = signal.blackmanharris(5)
bs = int(bs)

for ii in range(4):  # perform filters
    i = ii*bs

    # f_dc filter: Apply Blackman-Harris high pass filter to reduce very low frequency noise for waveform noise
    # the index of 0.5Hz is 25
    fourier = rfft(test_df.iloc[i:i+bs,1])
    for i in range(25):
        fourier[i] = fourier[i]*(window_10[i])
        fourier[i+int(bs/4)] = fourier[i+int(bs/4)]*(window_10[i])
   
    # f_50Hz filter: Apply Blackman-Harris notch filter to cut out 50Hz buzz noise and third harmonic
    # where index 2500 = 50Hz
    
    n = 2500
    fourier[n-28:n+27] = fourier[n-28:n+27]*(1-window_10)
    fourier[n-28+int(bs/4):n+27+int(bs/4)] = fourier[n-28+int(bs/4):n+27+int(bs/4)]*(1-window_10)
    #fourier[3*n-28:3*n+27] = fourier[3*n-28:3*n+27]*(1-window_10)
    #fourier[3*n-28+int(bs/4):3*n+27+int(bs/4)] = fourier[3*n-28+int(bs/4):3*n+27+int(bs/4)]*(1-window_10)
   
    # remove 100Hz + n * 100Hz as seen in batch 8
    # add up the 100Hz power. Just use positive frequencies for test but apply to pos and neg spectrum
    p100 = 0.
    n = 50
    for freq in range(100*n, 4900*n, 100*n):
        p100 += np.sum(abs(fourier[freq-5:freq+6] * window)**2)
    p = np.sum(abs(fourier[0:int(bs/4)])**2)
    #print('100Hz power in Batch ',ii+1, ' is ', p100, '. % of Total is ', p100/p)
    if p100/p >.001:
        print('batch ',ii+1, ' gets 100Hz filter')
        K = 5000
        sf_10 = irfft(fourier)
        sf = pd.DataFrame(sf_10, columns =['sf'])
        sf['sf'] = sf['sf'] - sf['sf'].shift(K)
        sf.loc[0:K-1,'sf'] = sf_10[0:K]
        sf_10 = sf.sf.values
        #for freq in range(100*n, 2000*n, 100*n):
            #fourier[freq-28:freq+27] = fourier[freq-28:freq+27]*(1-window_10)
            #fourier[freq-28+int(bs/4):freq+27+int(bs/4)] = fourier[freq-28+int(bs/4):freq+27+int(bs/4)]*(1-window_10)
    else:
        sf_10 = irfft(fourier)
    sfft_10 = np.append(sfft_10,sf_10)
        
test_df['signal_f_50Hz'] = 0.
test_df['signal_f_50Hz'] = sfft_10


names_in = ['signal','signal_f_50Hz']

features = pd.DataFrame()

for ii in range(4):# Produce engineered features from signal
    i = ii*bs
    feats, names_out = make_features(test_df.iloc[i:i+bs,[1,2]], names_in,4)
    features = pd.concat([features,feats], axis=0)
for i in names_out:
    test_df[i]=0.
test_df.loc[:,names_out] = features.loc[:,names_out].values


# In[ ]:




y_tree = grd.predict(features)

test_df['open_channels'] = y_tree

submit = test_df[['time','open_channels']]
submit.to_csv('submission.csv', index=False, float_format='%.4f')
del features, feats


# In[ ]:



examples = ['signal','open_channels', 'signal_n', 'signal_f_50Hz','50Hz_energy_floor','dt_50Hz_energy_floor']
fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 20))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(len(examples)):
    
    c = next(colors)["color"]
    ax[i].grid()
    if examples[i] in ['dt_50Hz_energy_floor','50Hz_energy_floor']:
        ax[i].plot(test_df['time'], test_df[examples[i]],color=c, linewidth= 2)
        ax[i].set_ylim(0,5)
    ax[i].plot(test_df['time'], test_df[examples[i]],color=c, linewidth=.5)
    ax[i].set_title(examples[i], fontsize=24)
    ax[i].set_xlabel('Time (seconds)', fontsize=18)
    #ax[i].set_ylabel('current (pA)', fontsize=24)
    #ax[i].set_ylim(-6,20)


# In[ ]:


bs = 500000
fs=10000.
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25, 20))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()
print("The four batches of test signal in the Frequency Domain")
print("The powerful spikes are identifiable noise. 50Hz is Euro power frequency noise.")
print("The central spike at f=0 is DC bias and low frequency waveform noise.")
for batch in range(10):
    fft = sp.fftpack.fft(train_df.signal_f_50Hz[bs*(batch):bs*(batch+1)])
    psd = np.abs(fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)
    
    i = abs(fftfreq) < 5000
    c = next(colors)["color"]
    ax[batch].grid()
    ax[batch].plot(fftfreq[i], np.log10(psd[i]),color=c, linewidth=.5)
    ax[batch].set_title(f'Batch {batch+1}')
    ax[batch].set_xlabel('Frequency (Hz)')
    ax[batch].set_ylabel('PSD (dB)')
    ax[batch].set_ylim(4,10)

