#!/usr/bin/env python
# coding: utf-8

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


import pyarrow.parquet as pq # convert parguet formatted files
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import *
import statsmodels.api as sm
from scipy import fftpack # Fast Fourier Transform functions


# In[ ]:


# plot settings
rand_seed = 135
np.random.seed(rand_seed)
xsize = 12.0
ysize = 8.0

from pylab import plot, show, savefig, xlim, figure,                 hold, ylim, legend, boxplot, setp, axes


# # Introduction
# Powerlines are measured using voltage, but you don't just measure voltage like you measure height. It's not just a single number. It goes up and down making waves. Each wave is a cycle and you can make judgements of each cycle by taking a multitude of measurements over time. For this competition, the underlying electric grid operates at 50 Hz (AKA 50 cycles per 1000 milliseconds). In other words, if we want to measure 50 cycles of voltage on this electric grid we would take measurements for 1000 milliseconds. The measurements in the competition were only performed for 20 milliseconds so we will only see one cycle. See basic math below:

# (50 cycles/ 1000 milliseconds) x (20 milliseconds) = 1 cycle

# In other words, the 800,000 measurements make up a time-course showing the voltage over time. 

# Now that we have that figured out, what is a 3-phase power scheme and what does that mean for us in this kaggle competition? To better understand, let's look at the data... 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_meta_df = pd.read_csv("../input/metadata_train.csv")\ntrain_df = pq.read_pandas("../input/train.parquet").to_pandas()')


# In[ ]:


train_meta_df.shape


# In[ ]:


train_meta_df.head(n=9)


# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# I just have book-keeping variables to help with transforming the voltage signals later...

# In[ ]:


# sampling rate
num_samples = train_df.shape[0] # 800,000 samples per signal
period = 0.02 # over a 20ms period
fs = num_samples / period # 40MHz sampling rate

# time array support
t = np.array([i / fs for i in range(num_samples)])

# frequency vector from FFT
freqs = fftpack.fftfreq(num_samples, d=1/fs)


# We have two training set files:

# * `metadata_train.csv` which contains four columns: 
#  * `signal_id`: a unique identifier so its meaningless
#   * `id_measurement`: ID code for each powerline. There should be 3 of each number which represents each phase in the 3-phase power scheme (AKA the trio)
#   * `phase`: the phase ID within the trio (0,1,2). 
#   * `target`: 0 fixed, 1 broken

# * `train.parquet` which contains the 800,000 rows for the 800,000 measurements for each respective `signal_id`

# Since each powerline has 3 rows of data (1 for each phase) in the `metadata_train.csv` and the `train.parquet` file that means the `target` is the same for each `signal_id` that has the same `id_measurement` (see plot below which shows the target variable for each phase seperately).  It also means we have 800,000 x 3 measurements per powerline.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfig, ax = plt.subplots()\nfig.set_size_inches(xsize, ysize)\n\nax =sns.countplot(x="phase", hue="target", data=train_meta_df, ax=ax)\nax.set_title("Distributions of `Target` variable for each phase is equal")\nplt.show()')


# We can also note from the plot above that there is a big target class imbalancement issue which could be dealt with via subsampling or loss functions, but I digress.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nplt.figure(figsize=(15, 10))\nplt.title("ID measurement:0, Target:0",\n         fontdict={\'fontsize\':36})\nplt.plot(train_df["0"].values, marker="o", label=\'Phase 0\')\nplt.plot(train_df["1"].values, marker="o", label=\'Phase 1\')\nplt.plot(train_df["2"].values, marker="o", label=\'Phase 2\')\nplt.ylim(-50,50)\nplt.legend()\nplt.show()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nplt.figure(figsize=(15, 10))\nplt.title("ID measurement:1, Target:1",\n         fontdict={\'fontsize\':36})\nplt.plot(train_df["3"].values, marker="o", label=\'Phase 0\')\nplt.plot(train_df["4"].values, marker="o", label=\'Phase 1\')\nplt.plot(train_df["5"].values, marker="o", label=\'Phase 2\')\nplt.ylim(-50,50)\nplt.legend()\nplt.show()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nplt.figure(figsize=(15, 10))\nplt.title("ID measurement:2, Target:0",\n         fontdict={\'fontsize\':36})\nplt.plot(train_df["6"].values, marker="o", label=\'Phase 0\')\nplt.plot(train_df["7"].values, marker="o", label=\'Phase 1\')\nplt.plot(train_df["8"].values, marker="o", label=\'Phase 2\')\nplt.ylim(-50,50)\nplt.legend()\nplt.show()')


# So based on our tiny sampling of these three powerlines it looks like amplitude could be a significant factor in predicting whether these powerlines are faulty.

# According to the wiki on the 3-phase power scheme:
# > In a symmetric three-phase power supply system, three conductors each carry an alternating current of the same frequency and voltage amplitude relative to a common reference but with a phase difference of one third of a cycle between each.

# Therefore, as competitors we need to look into smoothing techniques to measure the amplitude of these waves. We should also measure the variance of our estimated amplitude for each powerline across the 3 phases. 

# # Feature Engineering
# Since we don't want to do machine learning on the raw numbers from the parquet files (which could take forever and may not even be useful) I want to create features I can add to the meta tables to train on instead. We will also needs to add these features to the test set.

# Note that to make the code run faster I subset the training data for when I am editing this kernel but when I commit the code I do not do this...

# In[ ]:


# uncomment to subset the data (Note the graphs look really different when you do this)
#train_subset_df = train_df.iloc[:,range(0,99)]
#train_subset_meta_df = train_meta_df.iloc[range(0,99),:]

# uncomment to use the full dataset
train_subset_df = train_df
train_subset_meta_df = train_meta_df


# ## Mean, Median, and Standard Deviation of Measurements
# Multiple kernels have done this and found these numbers to be slightly useful so let's see...

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmean_list = train_subset_df.apply(np.mean)\nmedian_list = train_subset_df.apply(np.median)\nstd_list = train_subset_df.apply(np.std)')


# In[ ]:


mean_signal_df = mean_list.to_frame()
mean_signal_df = mean_signal_df.reset_index()
mean_signal_df = mean_signal_df.drop("index",axis=1)
train_subset_meta_df =train_subset_meta_df.merge(mean_signal_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"mean"})

median_signal_df = median_list.to_frame()
train_subset_meta_df =train_subset_meta_df.merge(median_signal_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"median"})

std_signal_df = std_list.to_frame()
train_subset_meta_df =train_subset_meta_df.merge(std_signal_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"std_dev"})


# In[ ]:


train_subset_meta_df.head(n=9)


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Mean Across Target Variables")
ax = sns.boxplot(x="target", y="mean", data=train_subset_meta_df)


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Median Across Target Variables")
ax = sns.boxplot(x="target", y="median", data=train_subset_meta_df)


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Stardard Deviation Across Target Variables")
ax = sns.boxplot(x="target", y="std_dev", data=train_subset_meta_df)


# ## Amplitude of Rolling Series
# Lets smooth the plots of the first 3 power lines to see if we can see any differences between these waves in target=0 verses target=1

# In[ ]:


ts1 = train_df["0"]
ts2 = train_df["1"]
ts3 = train_df["2"]

plt.figure(figsize=(16,6))
plt.title("ID measurement:0, Target:0",
         fontdict={'fontsize':36})
plt.plot(ts1.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts1.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts2.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts2.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts3.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts3.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.legend();


# In[ ]:


ts1 = train_df["3"]
ts2 = train_df["4"]
ts3 = train_df["5"]

plt.figure(figsize=(16,6))
plt.title("ID measurement:1, Target:1",
         fontdict={'fontsize':36})
plt.plot(ts1.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts1.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts2.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts2.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts3.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts3.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.legend();


# In[ ]:


ts1 = train_df["6"]
ts2 = train_df["7"]
ts3 = train_df["8"]

plt.figure(figsize=(16,6))
plt.title("ID measurement:2, Target:0",
         fontdict={'fontsize':36})
plt.plot(ts1.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts1.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts2.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts2.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.plot(ts3.rolling(window=100000,center=False).mean(),label='Rolling Mean');
plt.plot(ts3.rolling(window=100000,center=False).std(),label='Rolling sd');
plt.legend();


# It doesn't look like we can see any difference, but this is a small sample size to work with. Let's look at the amplitude across each target group. To calculate the amplitude, I smooth the powerline signals to create a single wave then I subtract the lowest and highest point. 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef calc_rolling_amp(row, window=100000):\n    return np.max(row.rolling(window,center=False).mean()) - np.min(row.rolling(window=100000,center=False).mean())\n\nrolling100k_amp = train_subset_df.apply(calc_rolling_amp)')


# In[ ]:


rolling100k_amp_df = rolling100k_amp.to_frame()
train_subset_meta_df =train_subset_meta_df.merge(rolling100k_amp_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"rolling100k_amp"})


# In[ ]:


train_subset_meta_df.head(n=9)


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Amplitude Across Target Variables")
ax = sns.boxplot(x="target", y="rolling100k_amp", data=train_subset_meta_df)


# It looks like there is not much of a difference unfortunately...

# 
# ## Measuring Amount of Noisy Points
# ### Number of points 1SD from the mean
# The next feature I am intersted in looking at is the number of data points in each signal that is greater than 1 SD from the mean.

# In[ ]:


def count1SDfromTheMean(row):
    max_1sd = np.mean(row) + np.std(row)
    min_1sd = np.mean(row) - np.std(row)
    noise_points = [x for x in row if (x > max_1sd) or (x < min_1sd)]
    return (len(noise_points))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncount1SDfromTheMean_list = train_subset_df.apply(count1SDfromTheMean)')


# In[ ]:


#%%time
#count1SDfromTheMean_list = train_df.apply(count1SDfromTheMean)


# In[ ]:


count1SDfromTheMean_df = count1SDfromTheMean_list.to_frame()
train_subset_meta_df =train_subset_meta_df.merge(count1SDfromTheMean_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"count1SDfromTheMean"})


# In[ ]:


train_subset_meta_df.head(n=9)


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Noise Count Across Target Variables")
ax =sns.boxplot(x="target", y="count1SDfromTheMean", data=train_subset_meta_df)


# There is not that much of a difference and spoiler alert the next parameter is probably better and gives basically the same information. I am going to drop this column.

# In[ ]:


# drop "count1SDfromTheMean" from the training set
train_subset_meta_df = train_subset_meta_df.drop(["count1SDfromTheMean"], axis=1)


# 
# ### Number of points 2SD from the mean
# The next feature I am intersted in looking at is the number of data points in each signal that is greater than 2 SD from the mean.

# In[ ]:


def count2SDfromTheMean(row):
    max_1sd = np.mean(row) + (2 * np.std(row))
    min_1sd = np.mean(row) - (2 * np.std(row))
    noise_points = [x for x in row if (x > max_1sd) or (x < min_1sd)]
    return (len(noise_points))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncount2SDfromTheMean_list = train_subset_df.apply(count2SDfromTheMean)')


# In[ ]:


count2SDfromTheMean_df = count2SDfromTheMean_list.to_frame()
train_subset_meta_df =train_subset_meta_df.merge(count2SDfromTheMean_df,"inner", 
                    left_index=True,right_index=True)
train_subset_meta_df = train_subset_meta_df.rename(index=str, columns={0:"count2SDfromTheMean"})


# In[ ]:


train_subset_meta_df.head()


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Noise Count Across Target Variables")
ax =sns.boxplot(x="target", y="count2SDfromTheMean", data=train_subset_meta_df)
plt.ylim(0,250)


# This looks like it could be a really useful feature!

# 1. ## Sum "extremeness" of each bin
# I want to try and bin the 800,000 data points and see if noise in a specfic location is significantly different in the different targets, but to do this I need to transform the waves so that they can be referenced equally. See kernel for explanation of how this transformation works:  https://www.kaggle.com/fernandoramacciotti/sync-waves-with-fft-coeffs

# In[ ]:


n_signals_to_load = 3
signals = pq.read_pandas(
    '../input/train.parquet', 
    columns=[str(i) for i in range(n_signals_to_load)]).to_pandas()
signals.columns


# In[ ]:


# get fft coeffs
def get_fft_coeffs(sig):
    return fftpack.fft(sig)

# get coeff with highest norm
def get_highest_coeff(fft_coeffs, freqs, verbose=True):
    coeff_norms = np.abs(fft_coeffs) # get norms (fft coeffs are complex)
    max_idx = np.argmax(coeff_norms)
    max_coeff = fft_coeffs[max_idx] # get max coeff
    max_freq = freqs[max_idx] # assess which is the dominant frequency
    max_amp = (coeff_norms[max_idx] / num_samples) * 2 # times 2 because there are mirrored freqs
    if verbose:
        print('Dominant frequency is {:,.1f}Hz with amplitude of {:,.1f}\n'.format(max_freq, max_amp))
    
    return max_coeff, max_amp, max_freq

# get max coeff phase
def get_max_coeff_phase(max_coeff):
    return np.angle(max_coeff)

# construct the instant angular phase vector indexed by pi, i.e. ranges from 0 to 2
def get_instant_w(time_vector, f0, phase_shift):
    w_vector = 2 * np.pi * time_vector * f0 + phase_shift
    w_vector_norm = np.mod(w_vector / (2 * np.pi), 1) * 2 # range between cycle of 0-2 
    return w_vector, w_vector_norm

# find index of chosen phase to align
def get_align_idx(w_vector_norm, align_value=0.5):
    candidates = np.where(np.isclose(w_vector_norm, align_value))
    # since we are in discrete time, threre could be many values close to the desired one
    # so let's take the one in the middle
    return int(np.median(candidates))


# In[ ]:


# align waves with np.roll()
align_phase = 0.5 # w_i = pi/2

fig = plt.figure(figsize=(12, 9))
plot_number = 0

for signal_id in signals.columns:
    # get samples
    sig = signals[signal_id]
    
    # fft
    fft_coeffs = get_fft_coeffs(sig)
    
    # asses dominant frequency
    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs, verbose=True)
    
    # phase shift
    ps = get_max_coeff_phase(max_coeff)
    
    # get angular phase vector
    w, w_norm = get_instant_w(t, f0, ps)
    
    # generate dominant signal at f0
    dominant_wave = amp * np.cos(w)
    
    # idx to roll
    origin = get_align_idx(w_norm, align_value=align_phase)
    
    # roll signal and dominant wave
    sig_rolled = np.roll(sig, num_samples - origin)
    dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)
    
    # plot signals
    plot_number += 1
    ax = fig.add_subplot(3, 1, plot_number)
    
    ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal
    ax.plot(t * 1000, dominant_wave_rolled, color='red', label='Rolled Wave at {:.0f}Hz'.format(f0)) # wave at f0
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal {} rolled'.format(signal_id))
fig.tight_layout()


# [](http://)What if you get the sum of the difference between the rolled original and rolledwave at 50 Hz in 125 bins?

# In[ ]:


# align waves with np.roll()
align_phase = 0.5 # w_i = pi/2
nbins=125
train_subset_meta_df.index = pd.RangeIndex(start=0, stop=len(train_subset_meta_df), step=1)


diff_df= pd.DataFrame()
for signal_id in train_subset_df.columns:
    # get samples
    sig = train_subset_df[signal_id]
    # fft
    fft_coeffs = get_fft_coeffs(sig)
    
    # asses dominant frequency
    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs, verbose=False)
    
    # phase shift
    ps = get_max_coeff_phase(max_coeff)
    
    # get angular phase vector
    w, w_norm = get_instant_w(t, f0, ps)
    
    # generate dominant signal at f0
    dominant_wave = amp * np.cos(w)
    
    # idx to roll
    origin = get_align_idx(w_norm, align_value=align_phase)
    
    # roll signal and dominant wave
    sig_rolled = np.roll(sig, num_samples - origin)
    dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)
    
    diff_bw_signAndDom = np.abs(dominant_wave_rolled-sig_rolled)
    sum_signals=[]
    numSignalsInBin=int(num_samples/nbins)
    #print(num_samples)
    #print(numSignalsInBin)
    for i in range(0,num_samples,numSignalsInBin):
        bin_sum = np.sum(diff_bw_signAndDom[i:i+numSignalsInBin])
        sum_signals.append(bin_sum)
    diff_df = diff_df.append(pd.Series(sum_signals), ignore_index=True)


# In[ ]:


colnamesOfDiffTable=["rolldiff"+str(x) for x in list(diff_df.columns)]
diff_df.columns = colnamesOfDiffTable
train_subset_meta_df =train_subset_meta_df.merge(diff_df,"inner", 
                    left_index=True,right_index=True)


# In[ ]:


train_subset_meta_df.head()


# In[ ]:


# function for setting the colors of the box plots pairs
def setBoxColorsOfTargets(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='orange')
    setp(bp['caps'][2], color='orange')
    setp(bp['caps'][3], color='orange')
    setp(bp['whiskers'][2], color='orange')
    setp(bp['whiskers'][3], color='orange')
    setp(bp['fliers'][2], color='orange')
    setp(bp['fliers'][3], color='orange')
    setp(bp['medians'][1], color='orange')


# In[ ]:


np.arange(0, 25, step=1)


# In[ ]:


for i in range(0,len(colnamesOfDiffTable),25):
    rolldiff_train_df = train_subset_meta_df.loc[:,["target"]+colnamesOfDiffTable[i:i+25]].copy()
    rolldiff_train_df = rolldiff_train_df.set_index("target")
    rolldiff_train_df = rolldiff_train_df.stack()
    rolldiff_train_df = rolldiff_train_df.reset_index()
    rolldiff_train_df.columns=["target","rolldiff","value"]

    plt.figure(figsize=(6,8))
    sns.set(style="whitegrid")
    plt.title("Sum of Extremeness across Bins in Target Variables")
    ax = sns.boxplot(x="rolldiff",y="value", hue="target", data=rolldiff_train_df)
    plt.xticks(np.arange(0, 25, step=1),np.arange(i, i+25, step=1))
    plt.xlabel("bin")
    plt.show()
    plt.close()


# In[ ]:


plt.figure(figsize=(6,8))
sns.set(style="whitegrid")
plt.title("Sum of Extremeness across Bins in Target Variables")
ax = sns.boxplot(x="target", y="std_dev", data=train_subset_meta_df)


#  Maybe next we can actually try and predict using these features...

# In[ ]:


train_subset_meta_df.to_csv('metadata_train_V2.csv')


# # Resources
# * https://www.kaggle.com/timothycwillard/vsb-power-line-faults-eda-feature-engineering
# * https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

# In[ ]:




