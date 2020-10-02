#!/usr/bin/env python
# coding: utf-8

# ![](http://)This notebook is a small dive into looking at the data provided for this comp.  The main focus is on looking at the EEG and ECG data.  The data was much noisier and difficult to interpret than I hoped, so this evaluation kind of went nowhere.  It was still interesting to look at some EEG, ECG, and RR data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from fastai.tabular import *
from pandas import *
from scipy import *
from matplotlib import gridspec
from scipy.fftpack import fft
from scipy.optimize import leastsq
from scipy import signal

import scipy.fftpack
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np


# # Download and browse
# 
# I have downloaded the data using the API via `kaggle competitions download -c reducing-commercial-aviation-fatalities`.  I have then extracted it and had to `chmod 666 train.csv` to get reading it working

# In[ ]:


df = pd.read_csv("../input/train.csv")


# Here we can see we have nearly 5million records, and the data looks pretty standard

# In[ ]:


len(df)


# In[ ]:


df.head()


# It was difficult to clearly see what the 'key' for the data is.  But my belief is that it is continuous for a crew member (`crew` + `seat`) within a single `experiment`.  This seems to make sensible data.
# 
# Each `experiment` has different `event` states

# In[ ]:


def get_data(crew, exp, seat):
    return df[(df.crew == crew) & (df.seat == seat) & (df.experiment == exp)].sort_values(by=['time'])

def trim_data(x, from_pt, secs):
    return x[(x.time >= from_pt) & (x.time < (from_pt + secs))]

crew_1_all = df[(df.crew == 1) & (df.seat == 0)].sort_values(by=['time'])
crew_1_ca = get_data(1, 'CA', 0)
crew_1_ss = get_data(1, 'SS', 0)
crew_1_da = get_data(1, 'DA', 0)


# Once again sanity checking the data, we now are down around 100k records for a single crew member in an experiment

# In[ ]:


len(crew_1_all), len(crew_1_ca), len(crew_1_ss), len(crew_1_da)


# Below we can see the various experiments use the same time periods (0 --> x) so it appears we have made the correct split

# In[ ]:


(crew_1_all.time.min(), crew_1_all.time.max()), (crew_1_ca.time.min(), crew_1_ca.time.max()), (crew_1_ss.time.min(), crew_1_ss.time.max()), (crew_1_da.time.min(), crew_1_da.time.max()),


# We can also see the event changes in each series, which appears to validate the assumption events happen periodically in an experiment.  However it is kinda disappointing that we basically have 1 event for both the CA & SS experiments

# In[ ]:


plt.plot(crew_1_ca.time, crew_1_ca.event);
plt.plot(crew_1_ss.time, crew_1_ss.event);
plt.plot(crew_1_da.time, crew_1_da.event);


# The time is in seconds, so the experiment seems to run for 360 seconds (6 minutes).  I'm now taking a sample from a random point in the data (in this case 120 seconds in) and taking a small sample (10 seconds) to try to look at the ECG data.
# 
# The ECG is the data I am most interested in analysing here

# In[ ]:


crew_to_use = crew_1_ca
secs = 10
crew_trim = trim_data(crew_to_use, 120.0, secs) # 10 secs of data from 2min in


# In[ ]:


plt.plot(crew_trim.time, crew_trim.ecg);


# This is nice validation, they said it is sampled at 256Hz and we have verified it here with the size of this sample.
# 
# Once again just looking at the data, it seems normal enough.  However there are some anomalies I have noticed, as e.g., below for crew 1 DA experiement there is no ECG...

# In[ ]:


plt.plot(crew_1_da.time, crew_1_da.ecg);


# In[ ]:


len(crew_trim), (len(crew_trim) / secs)


# In[ ]:


crew_trim.head()


# # ECG
# 
# Now I can plot the ECG for a 10 second period.  Given a normal HR (heart rate) is 60-100 bpm (beats per minute), we would expect between 10 and 17 beats in this window.  It should be easy to visual this count of beats.  If the pilot is stressed, we may see up to 30 beats in the period I guess, but it is a manageable amount

# As you can see in the plot, it is really hard to make out a standard QRS complex / heartbeat.  It looks close but just is all over the shop.  I reckon there is about 14 beats in this sample, with a lot of noise to clean up

# In[ ]:


plt.plot(crew_trim.time, crew_trim.ecg.rolling(8).mean());


# The interesting thing I have noticed with this is a bit of a wave in theh baseline, which seems to correlate with the respiratory rate as seen below.  ECGs should be a flatline, so I believe this really needs to be fixed up

# ## GSR Sidetrack
# 
# The GSR data looks a bit strange, continuously increasing? It doesn't seem correct, or at least not useful?

# In[ ]:


plt.plot(crew_trim.time, crew_trim.gsr);


# In fact, as seen below, the GSR data just seems a bit strange overall.  This article https://imotions.com/blog/gsr/ seems to suggest it should fluctuate a bit more consistently.  I will need to look into this further on its own

# In[ ]:


plt.plot(crew_to_use.time, crew_to_use.gsr);


# ## Resp Rate
# 
# The respiratory rate data looks pretty good - could calculate a rolling RR off this.  Perhaps detect the peak and record time since last/next breath?  We will do this later on.

# In[ ]:


plt.plot(crew_trim.time, crew_trim.r);


# ## Cleaning the ECG
# 
# Doing a larger rolling average sort of shows the same waves appearing in the ECG.  We need to put this through a low pass filter

# In[ ]:


plt.plot(crew_trim.time, crew_trim.ecg.rolling(256).mean());


# Here I do a really rough subtraction of the lower frequency MA from the higher frequency MA.  It seems to do what I want and level out the ECG, but I am still not happy with it.  The peaks are far too irregular and frequent - I believe further cleaning must be done.
# 
# Let's try proper fourier transforms & filtering, further in the analysis.  For now, I will leave this here.

# In[ ]:


cleared = crew_trim.ecg.rolling(8).mean() - crew_trim.ecg.rolling(256).mean()
plt.plot(crew_trim.time, cleared);


# # EEG
# 
# Having a look at the EEG data now, we have these fields
# 
# eeg_fp1
# eeg_f7
# eeg_f8
# eeg_t4
# eeg_t6
# eeg_t5
# eeg_t3
# eeg_fp2
# eeg_o1
# eeg_p3
# eeg_pz
# eeg_f3
# eeg_fz
# eeg_f4
# eeg_c4
# eeg_p4
# eeg_poz
# eeg_c3
# eeg_cz
# eeg_o2
# 
# EEG reference: https://en.wikipedia.org/wiki/Electroencephalography
# 
# Probe placement: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
# 
# Standard ordering?: https://en.wikipedia.org/wiki/Electroencephalography#/media/File:Human_EEG_without_alpha-rhythm.png
# 
# Brain layout: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)#/media/File:21_electrodes_of_International_10-20_system_for_EEG.svg
# 
# I have decided to follow this ordering: http://pediatrics.aappublications.org/content/pediatrics/129/3/e748/F2.large.jpg

# The plot below shows how noisy some of the EEG data is.  Once again I am plotting the moving average to smooth it out as per the ECG data.  I have not spent much time looking into better smoothing though

# In[ ]:


y0 = crew_trim['eeg_f8']
y1 = crew_trim['eeg_f8'].rolling(8).mean()
x = crew_trim.time

# graph it!
fig = plt.figure()
gs = gridspec.GridSpec(2, 1)

ax0 = plt.subplot(gs[0])
line0, = ax0.plot(x, y0, color='r')

ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(x, y1, color='b')

plt.setp(ax0.get_xticklabels(), visible=False)

yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

plt.subplots_adjust(hspace=.0)
plt.show()


# The code here is pretty rough, but just trying to replicate what an EEG printout looks like e.g., http://pediatrics.aappublications.org/content/pediatrics/129/3/e748/F2.large.jpg

# In[ ]:


ordered_fields = [['eeg_fp1', 'eeg_f3', 'eeg_c3', 'eeg_p3', 'eeg_o1'], 
                  ['eeg_fp2', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_o2'],
                  ['eeg_f7',  'eeg_t3', 'eeg_t5'],
                  ['eeg_f8',  'eeg_t4', 'eeg_t6'],
                  ['eeg_fz',  'eeg_cz', 'eeg_pz', 'eeg_poz']]

colours = ['darkslateblue', 'cadetblue', 
           'rebeccapurple', 'teal', 'chocolate']


# In[ ]:


fig = plt.figure(figsize=(12,12))
fld_cnt = sum([len(x) for x in ordered_fields])
gs = gridspec.GridSpec(fld_cnt, 1)

x = crew_trim.time
ax0 = None
axL = None

spines = ["top", "right", "left", "bottom"]

i = 0
for arr, col in zip(ordered_fields, colours):
    for fld in arr:
        ax = plt.subplot(gs[i])
        ln = ax.plot(x, crew_trim[fld].rolling(8).mean(), color=col)
        
        ax.set_yticks([])        
        ax.set(ylabel=fld.replace("eeg_", ""))
        ax.yaxis.label.set_rotation(0)
        ax.xaxis.grid(which="major")
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        
        for s in spines: ax.spines[s].set_visible(False)
        if i == 0: ax0 = ax
            
        axL = ax
        i = i+1

plt.setp(axL.get_xticklabels(), visible=True)
plt.subplots_adjust(hspace=.0)
plt.show()


# My uneducated opinion is that the plot looks pretty normal.  If you don't use the rolling average, it is very noisy.  I think basically all this data needs to be smoothed, but I have no idea wheter NN's and whatever frameworks can handle the data being that noisy.
# 
# You can see above how a lot of the signals are very similar.  I would expect to see them highly correlated in a correlation plot (e.g., like the dendogram from the fastai course)

# # Approximating the Respiratory Rate
# 
# Will try to work out the respiratory rate (RR) based on the signal we have.
# 
# Using some basic peak detection, we can see below we can easily find the peaks

# In[ ]:


x = crew_trim.r.values
avg = mean(x)
x = x - avg

peaks, _ = scipy.signal.find_peaks(x, distance=500)
peaks
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show();


# Now, we can easily look up the time of the peaks and calculate a rolling respiratory rate

# In[ ]:


peak_times = crew_trim.time.values[peaks]
peak_diffs = [peak_times[n] - peak_times[n-1] for n in range(1, len(peak_times))]
rolling_rr = [60/n for n in peak_diffs]

# Buff out the rolling RR as we don't have RR initially
# we just duplicate the initial value here, we could use zero if we wanted
rolling_rr.insert(0, rolling_rr[0])
peaks, peak_times, peak_diffs, rolling_rr


# Now we need to buff out the averages so it can stay consistent with the rest of the dataset.  We do 2 important things here that can be tweaked,
# 
# - Start with the first RR.  We could do something else if we wanted (e.g., 0)
# - Trail with the last values.  As above, we could do whatever here too

# In[ ]:


expanded_rr = [0] * len(x)
next_rr = 0
for i in range(0, len(x)):
    expanded_rr[i] = rolling_rr[next_rr]
    if (i == peaks[next_rr]) & (next_rr < (len(rolling_rr)-1)):
        next_rr = next_rr+1
plt.plot(crew_trim.time, expanded_rr);


# As we can see above, the RR in this slice is around 21-24 (per minute).  We can easily calculate this over the whole data set, and would provide a cleaner/simpler piece of data than the actual signals

# In[ ]:


def peak_rate(times, vals, d=None, p=None, w=None, h=None):
    avg = mean(vals[~np.isnan(vals)])
    vals = vals - avg
    peaks, _ = scipy.signal.find_peaks(vals, 
                                    distance=d, 
                                    prominence=p,
                                    width=w,
                                    height=h)
    peak_times = times.values[peaks]
    peak_diffs = [peak_times[n] - peak_times[n-1] for n in range(1, len(peak_times))]
    minute_rates = [60/n for n in peak_diffs]
    expanded_rates = [0] * len(vals)
    # Buff out the rolling rate as we don't have rate initially
    if len(minute_rates) > 0:
        minute_rates.insert(0, minute_rates[0])
        n = 0
        for i in range(0, len(vals)):
            expanded_rates[i] = minute_rates[n]
            if (i == peaks[n]) & (n < (len(minute_rates)-1)):
                n = n+1
    return pd.Series(expanded_rates)


# In[ ]:


def plot_rate(x, y, d=None, p=None, w=None, h=None):
    rates = peak_rate(x, y, d, p, w, h)
    # 7680 = 256*30 = average every 30 seconds?
    rolling_rate = rates.rolling(7680).mean()
    plt.plot(x, rates, color='lightgray');
    plt.plot(x, rolling_rate);


# In[ ]:


plot_rate(crew_to_use.time, crew_to_use.r, d=500)


# Above shows the resp rate for this crew_1 set, sitting around 22 the whole time.  Pretty high! but seems to be correct.  I would be feeding these numbers into a network, rather than the raw signals, as the RR is really what you are after.  Anything else in the RR signal is noise.
# 
# We could correlate this with events if desired.  This experiment we chose does not change event so it's probably pointless

# # ECG peak detection for heart rate
# 
# We will try the same method as the RR to calculate a RR.  The ECG data is very noisy though so we will use a rolling average immediately

# In[ ]:


x = crew_trim.ecg.rolling(48).mean().values #
avg = mean(x[~np.isnan(x)])
x = x - avg

peaks, _ = sp.signal.find_peaks(x, distance=64, prominence=1, width=1, height=50)
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
#plt.plot(np.zeros_like(x), "--", color="gray")
plt.show();


# The peak detection is picking up noise.  We can use the same function as the RR for the HR calc, as the method is identical.  We can see below that the HR is really high - over 150 constantly.  This is way too high, likely due to the noise.  However it seems very difficult to filter out the noise reliably.
# 
# Using the `.rolling(48).mean()` does not lower the HR either.

# In[ ]:


plot_rate(crew_to_use.time, crew_to_use.ecg, d=64, p=1, w=1)


# By setting the height it drops to more expected values, however it still seems far too noisy

# In[ ]:


plot_rate(crew_to_use.time, crew_to_use.ecg, d=64, p=1, w=1, h=50)


# # ECG Fourier Transforms
# 
# I tried various packages including wavelets and scipy to do a low-pass & high-pass filters of the ECG data, but could not get any to work.  I'd be interested to hear if anyone got it working.  Therefore, this section is still blank

# In[ ]:





# In[ ]:




