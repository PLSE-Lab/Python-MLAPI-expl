#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Introduction to physiological data
# 
# 
# 
# In this example, we're given three main physiological parameters:
# - Respiration
# - Electrocardiogram (ECG)
# - Electroencephalogram (EEG)
# 
# We'll talk through each of these in turn, some of their limitations, and how to process the data. I'm not going to talk about the galvanic skin response because it's use is, suffice to say, [controversial](https://sciencebasedmedicine.org/galvanic-skin-response-pseudoscience/).
# ## A short note on noise sources
# Biological sensors are quite susceptible to noise from outside sources. This can include lights (flickering at 50/60Hz depending on your AC frequency), and other electrical equipment. I think it's reasonable to assume that this experiment was in a chamber with a **tonne** of unshielded electronic high-tech stuff, all leaking noise at various frequencies. Hopefully this would be consistent between recordings, but it does make analysis more challenging, since removing any noise will usually remove a bit of signal too.
# 
# ## Respiration
# This is a simple measure of the rise and fall of the chest. It represents muscle activity of the diapragm and abdomen. We know that when someone is physiologically stressed, this rate increases. Could be interesting. 
# 
# Unfortunately, when we plot this data out, it seems to be largely affected by noise.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv('../input/train.csv')

# Just looking at a single trial for now
subset = df_train.loc[(df_train['crew'] == 1) & (df_train['experiment'] == 'CA')]

subset.sort_values(by='time')


# Show the plot
plt.plot(subset['r'][3000:4024])


# Oh dear, that's definitely not a normal respiration- you're looking at 5 seconds of data, which should show 1 or 2 breaths in a nice sinusoidal pattern. I think there's just too much high frequency noise for this to be useful.
# 
# Let's try some filtering to remove the high frequency signals
# 

# In[ ]:


from scipy import signal

b, a = signal.butter(8,0.05)

y = signal.filtfilt(b, a, subset['r'], padlen=150)

plt.plot(y[3000:4024])


# That's much better. So we should filter our data to get much more useful insights into it. We can then use clever libraries such as Biosppy to count the respiration rate, which is a more useful metric than the raw waveform.

# In[ ]:


from biosppy.signals import ecg, resp

out = resp.resp(y,sampling_rate=256, show=False)

plt.plot(out['resp_rate_ts'], out['resp_rate'])
plt.ylabel('Respiratory frequency [Hz]')
plt.xlabel('Time [s]');


# ## ECG
# This measures the electrical activity in the heart. This is a single lead ECG, useful for analysing the rhythm and rate. If someone had a heart condition, they'd be more likely to have a 12-lead ECG to look at the structural picture of any change in heart activity.<br>
# 
# ![The basic structure of the ECG](https://en.wikipedia.org/wiki/Electrocardiography#/media/File:SinusRhythmLabels.svg)
# 
# What's interesting about the ECG is while it's shape might vary between individuals or recordings (changes in lead position for instance), beat to beat, it's shape doesn't change much at all. The shape of the ECG can 'squash' down slightly as heart rate increases, but the amplitude is fixed, and really the only useful information for this experiment is the heart rate (which intuitively might be valuable as it increases when you're stressed).
# Here's the filtered ECG using the same settings as above

# In[ ]:


b, a = signal.butter(8,0.05)

y = signal.filtfilt(b, a, subset['ecg'], padlen=150)

plt.plot(y[3000:4024])


# If you want to convert this into heart rate data, we can use the template matching tool in Biosppy to detect the R waves, calculate their intervals, and work out the heart rate across the experiment. You might want to do some filtering on this to smooth out the heart rate, but the moment-to-moment heart rate is useful too.
# 

# In[ ]:



out = ecg.ecg(signal=subset['ecg'], sampling_rate=256, show=False)

plt.plot(out['heart_rate_ts'], out['heart_rate'])
plt.ylabel('Heart Rate (BPM)')
plt.xlabel('Time [s]');


# ## EEG
# Now this is the interesting bit to me. EEG's role has been greatly overstated over the years, and it's definitely not a panacea of brain activity. Clinically, you can usefully tell if someone is awake, asleep, brain dead, having a seizure, and a handful of other things. 
# EEG is a summation of all the electrical activity on the surface of the brain. This activity has to travel through layers of soft tissue, bone and skin, so it's no wonder that the data is quite noisy.
# ### Preparing EEG data
# This data is prepared in a fairly typical arrangement of 20 electrodes across the scalp. The letter in each lead signifies the part of the brain that that lead is nearest to (Temporal, Frontal, Parietal etc), with odd numbers on the left, evens on the right. Usually in the clinic, we don't look at the electrical potentials at each electrode, but at the potential difference between pairs of electrodes. This gives us an idea of the electrical field in the brain region between these two points as a way to infer what the brain is doing in that region. Clearly you can choose any two electrodes and produce 20! different potential differences, but not all of those are going to be useful. <br>
# We talk about the layout of choosing the pairs of electrodes to compare potential differences as **Montages**. There's lots of different montage systems, but commonly there's the 10-20 system. This data has an additional 'poz' electrode to the diagram, but that doesn't cause us a problem. <br>
# 
# ![10-20 Montage system](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/f2861bea35e87ac1fe5c053e4cc58911d28d112f/3-Figure1-1.png)
# 
# For this experiment, I chose the middle montage because it's one that's used clinically and I'm familiar with. Perhaps there's better ones for this experiment! <br>
# To montage the data, you just have to subtract the value of one electrode from another. It doesn't matter which way you do it, as long as it's consistent. I did this from front to back.

# In[ ]:


df_train['fp1_f7'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f7_t3'] = df_train['eeg_f7'] - df_train['eeg_t3']
df_train['t3_t5'] = df_train['eeg_t3'] - df_train['eeg_t5']
df_train['t5_o1'] = df_train['eeg_t5'] - df_train['eeg_o1']
df_train['fp1_f3'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f3_c3'] = df_train['eeg_f3'] - df_train['eeg_c3']
df_train['c3_p3'] = df_train['eeg_c3'] - df_train['eeg_p3']
df_train['p3_o1'] = df_train['eeg_p3'] - df_train['eeg_o1']

df_train['fz_cz'] = df_train['eeg_fz'] - df_train['eeg_cz']
df_train['cz_pz'] = df_train['eeg_cz'] - df_train['eeg_pz']
df_train['pz_poz'] = df_train['eeg_pz'] - df_train['eeg_poz']

df_train['fp2_f8'] = df_train['eeg_fp2'] - df_train['eeg_f8']
df_train['f8_t4'] = df_train['eeg_f8'] - df_train['eeg_t4']
df_train['t4_t6'] = df_train['eeg_t4'] - df_train['eeg_t6']
df_train['t6_o2'] = df_train['eeg_t6'] - df_train['eeg_o2']
df_train['fp2_f4'] = df_train['eeg_fp2'] - df_train['eeg_f4']
df_train['f4_c4'] = df_train['eeg_f4'] - df_train['eeg_c4']
df_train['c4_p4'] = df_train['eeg_c4'] - df_train['eeg_p4']
df_train['p4_o2'] = df_train['eeg_p4'] - df_train['eeg_o2']

features_n = ['fp1_f7', 'f7_t3', 't3_t5', 't5_o1', 'fp1_f3', 'f3_c3', 'c3_p3', 'p3_o1', 'fz_cz', 'cz_pz',
                'pz_poz', 'fp2_f8', 'f8_t4', 't4_t6', 't6_o2', 'fp2_f4', 'f4_c4', 'c4_p4', 'p4_o2', "ecg", "r", "gsr"]


# ### Analysing EEG data
# The interesting bit of EEG data comes from looking at the firing rate. With certain medical conditions, and in brain states, the neural activity starts to harmonise in pretty cool ways. The firing rate of this activity is measured in Hz, and grouped into bands:
# - Delta (<4Hz) Slow wave sleep, continous attention tasks
# - Theta (4-7Hz) Drowsiness, repression of elicited responses
# - Alpha (8-15Hz) Relaxed, eyes closed
# - Beta (16-31Hz) Active thinking, focus, alert
# - Gamma (>32Hz) Short term memory, cross sensory perception <br>
# 
# So, this looks easy, right? We just find out the firing rate of the EEG, if it's in Alpha or below, then we're happy that the pilot is either finding the task easy, or asleep. If it's Beta or above, then they're having a hard time focussing on their distraction. 

# In[ ]:


subset = df_train.loc[(df_train['crew'] == 1)]

# Discrete Fourier transform, using a hanning window of 1s
freqs, times, Sx = signal.spectrogram(subset['fz_cz'], fs=256, window='hanning', nperseg=256, noverlap=256-100, detrend=False, scaling='spectrum')
f, ax = plt.subplots(figsize=(12,5))
ax.pcolormesh(times, freqs, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]');


# There's definitely a change in frequency, but it's around 60Hz so it's probably an artefact (someone might have switched the light off), and there's nothing significant to see when you filter this out

# In[ ]:


b, a = signal.butter(8,0.2) 
y = signal.filtfilt(b, a, subset['fz_cz'], padlen=150)
freqs, times, Sx = signal.spectrogram(y, fs=256, window='hanning', nperseg=256, noverlap=256-100, detrend=False, scaling='spectrum')
f, ax = plt.subplots(figsize=(12,5))
ax.pcolormesh(times, freqs, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]');


# This is the problem, it's a noisy signal, and you can't pick out those rhythms easily. If anyone has any ideas to add to this please add to the discussion below!
