#!/usr/bin/env python
# coding: utf-8

# # Rock Music! Lomb-Scargle Periodograms for HF Noise Spectral Analysis
# ---
# 
# 
# As this competition will hinge largely on signal analysis, I thought I would make a feature engineering contribution that tries to isolate the main constituent frequencies amid the noise. 
# 
# Frequency analysis is typically performed with the [Fast Fourier Transform](http://mathworld.wolfram.com/FastFourierTransform.html). This breaks down a complex waveform into its sinusoidal constituents, mapping them into freqency space with their respective amplitudes:
# 
# ![](https://cdn.iopscience.com/images/books/978-1-627-05419-5/live/bk978-1-627-05419-5ch8f1_online.jpg)
# 
# Fourier transforms act on continuous periodic functions. Since real-life data is discretised, numerical solutions have to be found via methods like DFFT (Discrete Fast Fourier Transform). However, calculating this requires a time axis with regular intervals - that is, the time difference between successive amplitude measurements is constant. Many real-life situations, along with this competition, require frequency analysis on data where the time between measurements is unequal. What's the solution? The fantasically named **Lomb-Scargle Periodogram**:
# 
# ![](https://static.packt-cdn.com/products/9781785282287/graphics/B04223_06_11.jpg)
# 
# If this is new to you, please take a moment to say it again out loud. Great stuff! LSP is one approach among many for least-squares spectral analysis problems, and I'm sure other kagglers will find better solutions. But it's early days in the competition so hopefully this will help you get started! For a comprehensive primer on LSP, I strongly recommend [reading this article (PDF)](https://arxiv.org/pdf/1703.09824.pdf).
# 
# The test data doesn't include the time axis since that is our target, so when adding features you'll have to assume the time difference between measurements is even. With train.csv we can't make that assumption, which is why I tried this approach.
# 
# For speed, I'll be running this kernel on the data for the first earthquake period in train.csv only. I encourage other readers to expand on it for their own work - you might stumble on something critical to your final performance! As always, the first step is to read in the data and have a look at our variables:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import gc

#thanks to EEK: https://www.kaggle.com/theupgrade/creating-breaks-as-start-ends-of-bins
ttf_resets = np.array([5656575,50085879,104677357,138772454,187641821,
                  218652631,245829586,307838918,338276288,375377849,
                  419368881,461811624,495800226,528777116,585568145,
                  621985674]) - 1

train = pd.read_csv('../input/train.csv', nrows=ttf_resets[0])
#rename columns - quicker to type!
train.columns = ['signal', 'ttf']
print(train.describe())
train.plot(kind='scatter',x='ttf',y='signal', s=1)
plt.xlim(1.5, -0.05)
plt.title('Acoustic signal/ttf')
plt.show()


# The amplitude spikes during certain events. Periodic trends are difficult to identify visually. Taking a look at the signal amplitude distribution alone:

# In[ ]:


train.signal.hist(bins=200)


# A histogram of the entire signal data isn't particularly helpful since by nature, notable seismic activity manifests as high-amplitude burst phenomena. Removing the extreme outliers at TTF ~ 0.35s, we can get a clearer view of the majority of the data:

# In[ ]:


train.loc[(train.signal < 40) & (train.signal > -40), ].signal.hist(bins=200)


# An ordinary, symmetric distribution with a mean of around 4.56. I'm going to avoid mean normalisation since the general form of LSP doesn't require it, and a mean value != 0 is a feature in itself. Looking at the measurement times: 

# In[ ]:


train.ttf.hist()


# As expected, we have a mostly - but not perfectly - even time distribution. Python's inbuilt rounding limits how accurately we can see the see the time difference between data points but we can inspect them with the `decimal` package:

# In[ ]:


import decimal as dec
t1 = dec.Decimal(train.iloc[0, 1])
t2 = dec.Decimal(train.iloc[1, 1])
t3 = dec.Decimal(train.iloc[2, 1])
print('First TTF: ', t1, '(s)')
print('Second TTF: ', t2, '(s)')
print('Third TTF: ', t3, '(s)')

print('First delta t: ', t1-t2, '(s)')
print('Second delta t: ', t2-t3, '(s)')


# Delta-t for these adjacent readings is approximately 1.1E-9, meaning the [Nyquist Frequency](http://mathworld.wolfram.com/NyquistFrequency.html) - the upper bound on the frequency we can reliably detect - would be about 450MHz. For audio data, this is an extremely high frequency.
# 
# High-frequency waves are readily attenuated by rock, and actual seismological waves generally lie in the range of 0.1-1Hz since they travel further and hence have more predictive and forensic value. This experiment was at a much smaller scale however, so analysis of higher frequencies may yield valuable insights. So what is the *minimum* frequency we can reliably measure?
# 
# This poses something of a conundrum. The data is time-dependant, and ideally we'd like to have a number of LSP frequency spectra to examine as TTF approaches 0. This way, any changes in the signal profile over time can be examined - this is a prediction challenge after all. But the more samples we take, but smaller the timespan they will cover, and hence the lower bound for our frequency range gets higher. For this kernel I'm naively choosing a range of 10000 rows to get 566 different readings for the first earthquake, equivalent to around 0.0021 seconds per segment. For this the minimum frequency we could expect to differentiate would be approximately 1/0.021s or ~ 475Hz.
# 
# For future kernels I'll try a sliding-window approach with a larger number of samples to increase the frequency resolution and examine if low-frequencies have any notable trends in the training data. Be warned if doing this yourself as this is a very memory intensive process - running LSP on more than a couple of millions rows at once will give you the dreaded Memory Error! To start let's analyse the first sample alone:

# In[ ]:


from astropy.stats import LombScargle

t = train.ttf.values
y = train.signal.values

#astropy implementation works directly on frequency, not angular frequency
#autopower() calculates a frequency grid automatically based on mean time separation
frequency, power = LombScargle(t[0:10000], y[0:10000]).autopower(nyquist_factor=2)


# In[ ]:


plt.plot(frequency, power) 
plt.title('Noise LSP Frequency-Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()


# Overall the frequencies are highly smeared, characteristic of noisy data collected over time. A number of peaks are present but the amplitude is very low, far lower than anything one would consider to be statistically important. This is to be expected since this segment of the data doesn't display any notable acoustic activity. Taking a closer look at the initial peaks:

# In[ ]:


#power multiplied for graphical visibility
freq_df = pd.DataFrame({'freq': frequency.round(),
                       'amp': (power*1e6).round(1)})


# In[ ]:


import matplotlib.pyplot as plt
freq_df.loc[freq_df.freq < 1000000, :].plot(kind='scatter',x='freq',y='amp', s=2)
plt.title('Noise LSP Frequency-Amplitude Spectrum')
plt.show()


# In[ ]:


freq_df.loc[freq_df.freq < 150000, :].plot(kind='scatter',x='freq',y='amp', s=2)
plt.title('Noise LSP Frequency-Amplitude Spectrum')
plt.show()


# In frequency space, the acoustic noise displays sinusoidal trends along the 10kHz harmonics that are likely artefacts of the frequency grid created by `LombScargle`. Examining the frequencies sorted by amplitude:

# In[ ]:


freq_df.sort_values('amp', ascending=False).head(10)


# From the first 10 results, values that lie on the same frequency 'hump' can be seen with values close to eachother, which would yield unhelpful results. The initial method I'll be using to examine this data is as follows:
# 
# * Run LSP for each segment of 10000 rows up to TTF==0
# * Remove frequencies with an associated period smaller than the time window of these rows
# * Remove frequencies higher than a heuristically chosen value (optional)
# * Extract the 5 frequencies with the highest amplitudes **and** a minium separation of 0.2MHz and add them to train
# * Plot how these frequencies change in the build up to the quake
# * Examine how the extracted frequencies could be used to engineer new features 
# 
# This approach has its flaws. Notably, [as examined here](https://arxiv.org/pdf/1703.09824.pdf), taking the highest-amplitude frequency from LSP is a naive means of interpreting the data. The frequencies are highly spread out, so to remove values which are clustered on the same peak, adjacent frequency recordings have to be at least 0.2MHz apart. This corresponds to the approximate distance between 'humps'. Ideally we'd like to obtain one frequency for each 'hump' in the frequency spectrum, as are clearly visible below:

# In[ ]:


freq_df.loc[(freq_df.freq > 125000) & (freq_df.freq < 1000000), :].plot(kind='scatter',x='freq',y='amp', s=2)
plt.show()


# The quick and dirty function `arr_dist()` will run across an array and keep only values whose separation is greater than 0.2MHz until it has 5 values. The frequencies recorded here will be placed in 5 new columns in train. If expanding this process to other sections of the data, you may want to decrease the number of new columns since this particular segment of the data is rather small, and is less intensive to work with than most of the earthquakes.
# 
# This function is rather long but quite straightforward and easily modifiable. If you have any questions, I'll be happy to help in the comments.

# In[ ]:


def arr_dist(arr, sep, n=5):
    output = []
    for x in arr:
        keep=True
        for y in output:
            if abs(y-x)<sep:
                keep=False
                break
        if(keep):
            output.append(x)
            if len(output)==n:
                return(np.asarray(output))

ROWS_PER_SEGMENT = 10000
TIME_PER_SEGMENT = train.iloc[0, 1] - train.iloc[ROWS_PER_SEGMENT, 1]
MIN_FREQ = round(1/TIME_PER_SEGMENT)
MAX_FREQ = 1e8
FREQ_SEP = 0.2e6

def LSP_freq(df, signal_col='signal', time_col='ttf',
             nrows=ROWS_PER_SEGMENT, min_freq=MIN_FREQ, max_freq=MAX_FREQ):
    print('Lomb-Scargle Periodogram analysis commencing.')
    print('Minimum detection frequency: {}Hz'.format(MIN_FREQ))
    print('Manual maximum frequency cutoff: {}Hz'.format(MAX_FREQ))
    print('Number of segments: ', round(len(df)/ROWS_PER_SEGMENT))
    #initialise empty arrays for frequency outputs to be concatenated to DataFrame 
    freq_1 = np.zeros(len(df))
    freq_2 = np.zeros(len(df))
    freq_3 = np.zeros(len(df))
    freq_4 = np.zeros(len(df))
    freq_5 = np.zeros(len(df))
    segment_num = np.zeros(len(df))
    #loop through input DataFrame in chunks of length=nrows
    init_id = 0
    segment_id =1
    while init_id < len(df):
        if segment_id==1:
            print('Processing segment {:d}...'.format(segment_id))
        if segment_id%25==0:
            print('Processing segment {:d}...'.format(segment_id))
        end_id = min(init_id + nrows, len(df))
        ids = range(init_id, end_id)
        df_chunk = df.iloc[ids]
        #np arrays of amplitude and time columns
        signal = df_chunk[signal_col].values
        ttf = df_chunk[time_col].values
        #clear memory
        del df_chunk
        gc.collect()
        #calulate Lomb-Scargle periodograms for spectral analysis
        freq, amp = LombScargle(ttf, signal).autopower(nyquist_factor=2)
        freq_df = pd.DataFrame({'freq': freq.round(),
                               'amp': amp})
        #obtain frequencies sorted by highest amplitude as np.array
        top_freqs = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).freq.values
        del freq_df, freq, amp
        gc.collect()
        #obtain top 5 values that do not lie within 1kHz of eachother
        top_freqs = arr_dist(top_freqs, sep=FREQ_SEP)
        #sort principal frequencies from highest to lowest
        top_freqs = -np.sort(-top_freqs)
        #update main frequency component arrays
        freq_1[ids] = top_freqs[0]
        freq_2[ids] = top_freqs[1]
        freq_3[ids] = top_freqs[2]
        freq_4[ids] = top_freqs[3]
        freq_5[ids] = top_freqs[4]
        segment_num[ids] = segment_id
        del top_freqs
        init_id += nrows
        segment_id += 1
    print('...Done. Adding main component frequencies as DataFrame columns...')
    df['Freq_1'] = freq_1
    df['Freq_2'] = freq_2
    df['Freq_3'] = freq_3
    df['Freq_4'] = freq_4
    df['Freq_5'] = freq_5
    df['Segment'] = segment_num
    df['Freq_MinMax'] = df['Freq_1'] - df['Freq_5']
    print('...Done.')
    
LSP_freq(train)


# In[ ]:


train.head()


# Now we can examine the noise component frequencies over time, smoothing them via a rolling-mean:

# In[ ]:


cols = ['Freq_1', 'Freq_2','Freq_3','Freq_4','Freq_5']
for x in cols:
    train[x + '_RM'] = train[x].rolling(window=100000,center=False).mean()
cols_rm = []
for x in range(0, 5):
    cols_rm.append(cols[x] + '_RM')

ax = plt.gca()
train.plot(kind='line',x='ttf',y=cols_rm ,ax=ax, figsize=(12, 6))
plt.xlim(1.5, -0.05)
plt.title('5 Main Noise Component Frequencies (LSP, min separation 200kHz)')
plt.xlabel('TTF(s)')
plt.ylabel('Frequency (Hz)')
plt.show()


# There's a lot to examine here, and certainly many parameters to change that may lead to more informative results. The main frequencies spike during the earthquake event, but this is not the only time they do so. This data is independent of the signal amplitude, and seems to be sensitive to smaller jumps in amplitude that also cause high-frequency noise. 

# In[ ]:


train['FREQ_MEAN'] = train[cols].mean(axis=1)
train['FREQ_MEAN_RM'] = train['FREQ_MEAN'].rolling(window=250000, center=False).mean()
train.plot(kind='line',x='ttf',y='FREQ_MEAN_RM', figsize=(12, 6))
plt.xlim(1.5, -0.05)
plt.show()


# In[ ]:


train['FREQ_RATIO'] = train['Freq_1']/train['Freq_2']
train['FREQ_RATIO_RM'] = train['FREQ_RATIO'].rolling(window=500000, center=False).mean()
train.plot(kind='line',x='ttf',y='FREQ_RATIO_RM', figsize=(12, 6))
plt.xlim(1.5, -0.05)
plt.title('Ratio of Highest Main Frequency/Second Highest: Rolling-Mean')
plt.show()


# Determining trends visually is difficult, which is no surprise since the data by nature is both stochastic and noisy. Also, will identified trends be reapeated in the data for other earthquakes?
# 
# To isolate instances where the frequency detected by LSP is statistically viable, a typical approach is a false-alarm filter on the results, from [Scargle's original 1982 paper](https://www.researchgate.net/profile/Thomas_Ruf3/publication/245535651_The_Lomb-Scargle_Periodogram_in_Biological_Rhythm_Research_Analysis_of_Incomplete_and_Unequally_Spaced_Time-Series/links/02e7e51d72fa498ab1000000.pdf): ![](https://i.imgur.com/O0ByWdL.png)  
# 
# This approach has not been successful during my work, and has been [identified as an inappropriate method](https://github.com/astropy/astropy/issues/7618) for assessing the viability of identified frequencies. In any case, the nature of this dataset limits the occurance of statistically notable frequencies. Instead I'll be using an amplitude cut-off which displays the occurances of frequencies that, while not having a p-value < 0.05, still stand out from the background noise.

# In[ ]:


#false-alarm filter - unused in this kernel
def LSP_filter(p, M):
    threshold = -np.log(1-((1-p)**(1/M)))
    return(threshold)


# In[ ]:


ROWS_PER_SEGMENT = 1000
TIME_PER_SEGMENT = train.iloc[0, 1] - train.iloc[ROWS_PER_SEGMENT, 1]
MIN_FREQ = round(1/TIME_PER_SEGMENT)
MAX_FREQ = 1e10
THRESHOLD = 0.1

def LSP_freq_filtered(df, signal_col='signal', time_col='ttf',
             nrows=ROWS_PER_SEGMENT, min_freq=MIN_FREQ, max_freq=MAX_FREQ):
    print('Lomb-Scargle Periodogram analysis commencing.')
    print('Minimum detection frequency: {}Hz'.format(MIN_FREQ))
    print('Manual maximum frequency cutoff: {}Hz'.format(MAX_FREQ))
    #initialise empty arrays for frequency outputs to be concatenated to DataFrame 
    freq_1 = np.zeros(len(df))
    amps_1 = np.zeros(len(df))
    segment_num = np.zeros(len(df))
    #loop through input DataFrame in chunks of length=nrows
    init_id = 0
    segment_id =1
    while init_id < len(df):
        if segment_id==1:
            print('Processing segment {:d}...'.format(segment_id))
        if segment_id%500==0:
            print('Processing segment {:d}...'.format(segment_id))
        end_id = min(init_id + nrows, len(df))
        ids = range(init_id, end_id)
        df_chunk = df.iloc[ids]
        #np arrays of amplitude and time columns
        signal = df_chunk[signal_col].values
        ttf = df_chunk[time_col].values
        #clear memory
        del df_chunk
        gc.collect()
        #calulate Lomb-Scargle periodograms for spectral analysis
        freq, amp = LombScargle(ttf, signal).autopower(nyquist_factor=2)
        freq_df = pd.DataFrame({'freq': freq.round(),
                               'amp': amp})
        freq_df = freq_df.loc[freq_df['amp'] >= THRESHOLD] 
        #obtain frequencies sorted by highest amplitude as np.array
        top_freqs = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).freq.values
        amps = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).amp.values
        del freq_df, freq, amp, signal, ttf
        gc.collect()
        #update main frequency component arrays
        try:
            freq_1[ids] = top_freqs[0]
            amps_1[ids] = amps[0]
        except:
            freq_1[ids] = 0
        segment_num[ids] = segment_id
        del top_freqs
        init_id += nrows
        segment_id += 1
    print('...Done. Adding main component frequencies as DataFrame columns...')
    df['Freq_periodic'] = freq_1
    df['Amps_periodic'] = amps_1
    df['Segment'] = segment_num
    print('...Done.')
    
LSP_freq_filtered(train)


# In[ ]:


train.head()


# In[ ]:


train['Freq_RM'] = train['Freq_periodic'].rolling(window=200000,center=False).std()
train.plot(kind='scatter',x='ttf',y='Freq_RM', figsize=(12, 6), s=1)
plt.xlim(1.5, -0.05)
plt.axvline(0.32)
plt.text(0.3,2e8,'Main Quake')
plt.title('Lomb-Scargle Main Component Frequency Standard Deviation (100MHz): Rolling-Mean ')
plt.show()


# In[ ]:


train['Amp_RM'] = train['Amps_periodic'].rolling(window=200000,center=False).mean()
train.plot(kind='scatter',x='ttf',y='Amp_RM', figsize=(12, 6), s=1)
plt.xlim(1.5, -0.05)
plt.title('Lomb-Scargle Main Component Frequency Amplitude: Rolling-Mean')
plt.axvline(0.32)
plt.text(0.3,0.01,'Main Quake')
plt.show()


# These are potentially significant results for predicting a TTF < 0.2 seconds. Following an earthquake event, the statistical randomness of the noise increases, so the maximum amplitude for the main frequency detected by LSP drops below the mean. Meanwhile, the standard deviation of the main frequency isolated by LSP decreases. Physical reasons for this may include a decrease in the acoustic energy, and an increase in entropy following the rapid release of tension in a system. I'll leave that to the researchers to explain! 
# 
# There are other features that also decrease as TTF approaches 0, so nothing in this kernel will be a magic bullet. But there may still be some useful features that can be engineered from LSP frequency analysis, and I hope this kernel helps my fellow competitors. Even if these conclusions aren't helpful to you, there is much for you to build on.Good luck! 
