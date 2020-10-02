#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
print(os.listdir("../input"))
PATH = '../input/'
# Any results you write to the current directory are saved as output.
np.random.seed(512)


# # Earthquake prediction 
# 
# Well my first kernel, and just some work in progress (many a fast evening hacks ...)
# 
# Idea collection and some stupid(?) modeling ideas.
# 
# I had a look before at  https://www.kaggle.com/jsaguiar/seismic-data-exploration by aguiar, who points out many things, that could be an issue for this kernel... And I will sometimes refer to the quite nice analysis in the kernel. 
# 
# 
# First things first:
# 1. We got one big training file
#     * It's possible a concatenation of different experiments (see other kernels, different time steps?
# 2. We got many small test sets.
#     * Of 150.000 data points of acoustic data
# 
# Let's have a look at it.
# To lower the memory burden a little bit, I am only reading every second row for now. Not sure about the effect on the data yet. (Doesn't work on the kernel, higher memory load, than without...)
# 
# ## TODO:
# Get that timing clear!

# In[ ]:


# https://nikgrozev.com/2015/06/16/fast-and-simple-sampling-in-pandas-when-loading-data-from-files/ 
# Following the approach to index the rows in pandas skiprows. Trust me with the lenght or run:
# train_lines = sum(1 for l in open(f'{PATH}train.csv'))
train_lines = 629145481 - 1 # there needs to be some substraction (0 indexing)
# if not using pandas with skiprows:
#skridx = np.arange(0, train_lines, 2)
samp_length = 75000


# In[ ]:


# I used the dtype options from aguiars kernel, as this apparently solved the memory overflow in kaggle kernel
train = pd.read_csv(f'{PATH}train.csv', dtype={'acoustic_data': np.int16, 
                                               'time_to_failure': np.float64}) # something less than 5gb, actually - things were getting worse with skiprows, so now ... 
test_ids = pd.read_csv(f'{PATH}sample_submission.csv') 

# Well decimating now here:


# In[ ]:


# skridx = np.arange(0, train_lines, 2) # Decimate and hoping not to kill the memory again. 
train = train.iloc[::2, :]


# In[ ]:


# Short overview over the file format, note everything is decimated a bit:
train.head()


# In[ ]:


# Many many samples
train.shape


# In[ ]:


# We need to predict the time_to failure for simple snippets.
test_ids.head()


# In[ ]:


# 2624 test files.
test_ids.shape 


# Immediate question before going on with the training data: What is the length of each test set?
# ``test_length = [pd.read_csv(f'{PATH}test/{i}.csv').shape[0] for i in test_ids['seg_id']]``
# 
# Well ... it's 150000, which can be found in many of the other kernels (i.e. aguiars). 
# Running the line above would be a waste of time :P 

# # Data overview

# In[ ]:


train['time_to_failure'].describe()


# In[ ]:


train['acoustic_data'].describe()


#  Next we have a look at another, question which is interesting for me: What is the sampling frequency. 
#  How many data points are in the acoustic data before a change in time_to_failure occurs?

# In[ ]:


unique_sampling_steps = np.unique(np.diff(np.round(train.time_to_failure, 9)))


# In[ ]:


# We know there are several events. So a slightly different plot:
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(unique_sampling_steps)
plt.title('Steps with events')
plt.subplot(122)
plt.hist(unique_sampling_steps[unique_sampling_steps < 2])
plt.title('Removing the long events.')
plt.tight_layout()


# But apparently, there are different  sampling periods (see the referred kernel...).  
# We can also see this above.
# 
# So maybe later: Discretize the data a bit more (well here already half the data was skipped anyways...)
# 
# Stupid other idea: Strict discretization and clustering? 

# In[ ]:


print(np.where(np.diff(np.round(train.time_to_failure[:100000], 5)))) # I.e. decreasing precision 


# In[ ]:


# Now lets have a look at the audio signal around failures, with the index we are not really caring about being one off..
# looking for increasing times as there's a restart in the signal
failures = np.where(np.diff(train.time_to_failure) > 0)


# In[ ]:


# Seems like there are 16 failures:
print(failures[0].shape)


# ## Looking at the 150000 samples before each event:

# In[ ]:


plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    plt.subplot(4, 4, ii + 1)
    plt.plot(np.arange(-samp_length, 100), train.acoustic_data[f_idx - samp_length : f_idx + 100])


# ## And look at some more data before events:
# Well this is not very informative /  representative.  

# In[ ]:


plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    plt.subplot(4, 4, ii + 1)
    plt.plot(np.arange(-1200000, 0), train.acoustic_data[f_idx - 1200000 : f_idx])


# Kinda smooth plots before each event, not high value data. Just by eyeballing it seems like that there is a decay in high values, the closer we are to an earthquake. But well, this is not the best way to look at the data. 
# 
# Additional point: We are not able to rely on too many data points before events to make our predictions. Rather (as the goal of the challenge is) we have to create a model to just predict the time before an event. 
# Cutting into samples could solve this maybe.
# 
# We also see that there are huge spikes in the data!

# # Learn a bit more about time to failure

# Our goal is to predict one value for time_to_failure for each training segment. A question that came to my mind, also due to the different sampling rates: How much does time_to_failure vary in random 150.000 (or 75.000) sample pieces. 

# In[ ]:


n_rnd_samples = 1000
rnd_sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_rnd_samples)


# In[ ]:


variation = np.empty(n_rnd_samples)

for idx, samp in enumerate(rnd_sample_idx):
    variation[idx] = np.max(train.time_to_failure[samp - samp_length : samp]) - np.min(train.time_to_failure[samp - samp_length : samp])


# ## Actually - not so much

# In[ ]:


b = plt.hist(variation[variation < 4])


# # Learn a bit more about acoustic data
# 
# ## Spikes
# 
# We have seen some huge spikes in the data. Are these common?

# In[ ]:


b = plt.hist(np.clip(train.acoustic_data, -200, 200), bins=50)


# Apparently not

# ## Can Spikes tell us something about time_to_failure?
# Only using absolute values for spikes

# In[ ]:


from scipy.stats import spearmanr
spike_size = 800
spike_idx = np.abs(train.acoustic_data) > spike_size

plt.figure(figsize=(15,5))
spear = spearmanr(np.abs(train[spike_idx]['acoustic_data']), train[spike_idx]['time_to_failure'])
plt.subplot(121)
plt.scatter(np.abs(train[spike_idx]['acoustic_data']), train[spike_idx]['time_to_failure'])
b = plt.title(f'Peaks and time to failure correlate with r ={spear[0]:.3f}')
plt.subplot(122)
b = plt.hist(train[spike_idx]['time_to_failure'])
plt.tight_layout()


# We can see that there seems to be some relationship between spikes and time to failure.
# Huge spikes seem to occur somewhat more readily before failure events!
# 
# But also caution: Some spikes occur long before!

# ## Do some random sampling again: Look at the max-values in an epoch and time_to_failure

# In[ ]:


t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty(n_rnd_samples)

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx] = np.max(np.abs(train.acoustic_data[samp - samp_length : samp]))


# In[ ]:


spear = spearmanr(np.array(t_t_f), np.array(m_a_d))

plt.scatter(np.array(t_t_f), np.array(m_a_d))
b = plt.title(f'Peaks and time to failure correlate with r = {spear[0]:.3f}')


# We can actually see, that some really basic features (such as the max-value) has some predictive ability!

# # Extend the notion - can we use basic properties of the distribution?

# In[ ]:


def basic_properties(data):
    properties = np.zeros(7)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = list(np.percentile(np.abs(data), [25, 50, 75]))
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, np.min]):
        properties[3 + n_f] = jj(data)
    return properties
    
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty((n_rnd_samples, 7))

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, :] = basic_properties(np.abs(train.acoustic_data[samp - samp_length : samp]))


# In[ ]:


plt.figure(figsize=(25,20))
feat_names = ['25-percentile', '50-percentile', '75-percentile', 'mean', 'std', 'max', 'min']
for ii, fna in enumerate(feat_names):
    ax = plt.subplot(3, 4, ii + 1)
    sns.regplot(t_t_f, m_a_d[:, ii], ax=ax)
    spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, ii]))
    b = plt.title(f'{fna} and ttf correlate with r = {spear[0]:.3f}')


# Another value I just remembered that is often used in the auditory domanin ist the RSM (root-squared-mean). Which should correlated highly with values in the above. But I'll but it in here as well. 

# In[ ]:


t_t_f = np.empty(n_rnd_samples)
m_a_d = np.hstack([m_a_d, np.zeros((m_a_d.shape[0],1))])

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, 7] = np.sqrt(np.mean(train.acoustic_data[samp - samp_length : samp] ** 2))
    


# In[ ]:


plt.figure(figsize=(15,5))
ax = plt.subplot(1, 2, 1)
sns.regplot(t_t_f, m_a_d[:, 7], ax=ax)
spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, 7]))
b = plt.title(f'RMS and ttf correlate with r = {spear[0]:.3f}')
ax = plt.subplot(1, 2, 2)
sns.regplot(m_a_d[:,3], m_a_d[:, 7], ax=ax)
spear = spearmanr(m_a_d[:,3], np.array(m_a_d[:, 7]))
b = plt.title(f'RMS and abs(mean) correlate with r = {spear[0]:.3f}')


# However, it might be interesting to check whether normalizing the data to have the same RMS will have an effect on predictive ability (generalization etc.)

# ## Punchline:
# Some basic features could actually do something already! We tried this already, see below Version 7, (LB=1.962)

# # Addendum 1
# Using amplitude, power and other simple features deal with the data at hand directly. Another important feature of the audio signal could be the change over time. That is calculating the derivative, but now using the simple difference over time.  
# **Important now**: The decimation step in the beginning is now affecting the data!

# In[ ]:


# This might use up a lot of memory... 
train['acoustic_diff'] = np.hstack([0, np.diff(train['acoustic_data'])])


# In[ ]:


# simple description:
train['acoustic_diff'].describe()


# ## Side note
# I am not plotting the time series before the events, as those look pretty similar to the normal data. So there is not too much information to be gained by this. 

# ## Redoing the basic features analysis:

# In[ ]:



def RMS(data):
    return np.sqrt(np.mean(data**2))

def basic_properties_up(data):
    properties = np.zeros(8)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = np.percentile(data, [25, 50, 75])
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, np.min, RMS]):
        properties[3 + n_f] = jj(data)
    return properties
    
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty((n_rnd_samples, 8))

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, :] = basic_properties_up(train.acoustic_diff[samp - samp_length : samp])


# In[ ]:


plt.figure(figsize=(25,20))
feat_names = ['25-percentile', '50-percentile', '75-percentile', 'mean', 'std', 'max', 'min', 'RMS']
for ii, fna in enumerate(feat_names):
    ax = plt.subplot(3, 4, ii + 1)
    sns.regplot(t_t_f, m_a_d[:, ii], ax=ax)
    spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, ii]))
    b = plt.title(f'{fna} and ttf correlate with r = {spear[0]:.3f}')


# Some basic properties of change also correlate with time to failure. But without further analysis we cannot be sure, how unique those properties are! 

# # Addendum 2: time-frequency features

# Some common features that are used in the auditory domain are time frequency estimates. I am pretty much agnosticly applying those here. I have some ideas how to implement them more nicely, especially for visualization, but again, some evening hack... 

# In[ ]:


import librosa # Seems to be a common library for all thing auditory


# A feature quite often used are mel frequencies. Do they work here? I don't know. Very often used in speech processing!
# Working with defaults here. But let's see what happens in another domain, right before an event.
# But we need a sampling rate here... 

# In[ ]:


# Assuming that t_t_f is in seconds, yes, I'm lazy, still didn't look that up, shame on me:
sr = round(1 / (train.iloc[0]['time_to_failure'] - train.iloc[1]['time_to_failure']))
print(sr)


# Still not having much of an idea about earthquake data, that seems to be a quite high sr. 

# In[ ]:


(train.iloc[0]['time_to_failure'] - train.iloc[75000]['time_to_failure'])


# And something is wrong. Let's just use the defaults and see how far we can get

# In[ ]:


def mfcc_wrap(data):
    return librosa.power_to_db(librosa.feature.melspectrogram(data.values.astype('float32'), n_mels=25))


# # Hm... some look at db in a 75000 sample

# In[ ]:


plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    ax = plt.subplot(4, 4, ii + 1)
    sns.heatmap(mfcc_wrap(train.acoustic_data[f_idx - samp_length : f_idx + 100]))


# In[ ]:


mfcc_wrap(train.acoustic_data[ : samp_length]).ravel().shape


# Not really sure what to make off it... yet... this will happen another time...

# # Let's try some really simple prediciton
# In the prior version there was stuff done using fft features (which was probably quite wrong on many levels), it didn't do much too say the least^^
# 
# By looking at the other analysis, it was probably a relation between max values and time_to_failure

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# In[ ]:


max_features = np.round(train.shape[0]/ samp_length).astype(int)
print(f'We can create {max_features} unique samples. Maybe this gets us somewhere')


# In[ ]:


def basic_properties_time(data):
    properties = np.zeros(7)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = np.percentile(data, [25, 50, 75])
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, RMS]):
        properties[3 + n_f] = jj(data)
    return properties


# In[ ]:


def basic_properties_diff(data):
    properties = np.zeros(6)
    properties[:2] = np.percentile(data, [25, 75])
    
    for n_f, jj in enumerate([np.min, np.std, np.max, RMS]):
        properties[2 + n_f] = jj(data)
    return properties


# ## Some things that might help
# We can create 4194 unique samples. But I think random sampling from the data to create our test and validation sets might be the way to got. Let's see how far we can get! 

# In[ ]:


train =  train.drop(['acoustic_diff'], axis=1)


# In[ ]:


n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))


# In[ ]:


sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    X_time[idx, :] = basic_properties_time(np.abs(train.acoustic_data[samp - samp_length : samp]))
    X_diff[idx, :] = basic_properties_diff(np.diff(train.acoustic_data[samp - samp_length : samp]))
    X_mfcc[idx, :] = mfcc_wrap(train.acoustic_data[samp - samp_length : samp]).ravel()


# In[ ]:


pred_mean = np.ones(y.shape) * np.mean(X_time[:, 3])


# # Predictions:

# ## Base line - what we definitely want to beat!

# In[ ]:


print(mean_absolute_error(y, pred_mean))


# ## Trees have been promising before

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Make pred_tree greater than 0
pred_time = cross_val_predict(RandomForestRegressor(n_estimators=100), X_time, y, cv=3)
pred_time[pred_time < 0] = 0


# In[ ]:


print(mean_absolute_error(y, pred_time))


# At least better than baseline... 

# ## Time derivative features:

# In[ ]:


pred_diff = cross_val_predict(RandomForestRegressor(n_estimators=100), X_diff, y, cv=3)
pred_diff[pred_diff < 0] = 0


# In[ ]:


print(mean_absolute_error(y, pred_diff))


# A bit better still ...

# In[ ]:


# simple average:
print(mean_absolute_error(y, (pred_time + pred_diff)/2))


# ## Time and derivatives combined

# In[ ]:


pred_comb = cross_val_predict(RandomForestRegressor(n_estimators=100), np.hstack([X_time, X_diff]), y, cv=3)


# In[ ]:


pred_comb[pred_comb < 0] = 0
print(mean_absolute_error(y, pred_comb))


# #### Bottom line:
# Time derivative features seem to contain at least some more information

# ## A quick look a the mel frequencies

# In[ ]:


pred_mel = cross_val_predict(RandomForestRegressor(n_estimators=10, n_jobs=3), X_mfcc, y, cv=3)


# In[ ]:


pred_mel[pred_mel < 0] = 0
print(mean_absolute_error(y, pred_mel))


# Something more has to be done here! ... Probably understanding wise... 

# ## And the average

# In[ ]:


print(mean_absolute_error(y, (pred_mel + pred_comb)/2))


# # A last test - does normalization of batches help?

# In[ ]:


data_rms = RMS(train.acoustic_data)
print(data_rms)


# ## We are now recreating the data set and look at a first basic preprocessing step

# Basically we are rescaling every batch in the training set by the factor data_rms / set_rms

# In[ ]:


n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))


# In[ ]:


sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    temp_data = train.acoustic_data[samp - samp_length : samp]
    temp_rms = RMS(temp_data)
    temp_data = temp_data * (data_rms/temp_rms)
    
    X_time[idx, :] = basic_properties_time(np.abs(temp_data))
    X_time[idx, -1] = temp_rms # keeping the old rms as a feature basic_properties_time(np.abs(temp_data))
    X_diff[idx, :] = basic_properties_diff(np.diff(temp_data))
    X_mfcc[idx, :] = mfcc_wrap(temp_data).ravel()


# ## The rescaled error values:

# In[ ]:


pred_time = cross_val_predict(RandomForestRegressor(n_estimators=100), X_time, y, cv=3)
pred_time[pred_time < 0] = 0
print(mean_absolute_error(y, pred_time))


# In[ ]:


pred_diff = cross_val_predict(RandomForestRegressor(n_estimators=100), X_diff, y, cv=3)
pred_diff[pred_diff < 0] = 0
print(mean_absolute_error(y, pred_diff))


# In[ ]:


pred_comb = cross_val_predict(RandomForestRegressor(n_estimators=100), np.hstack([X_time, X_diff]), y, cv=3)
pred_comb[pred_comb < 0] = 0
print(mean_absolute_error(y, pred_comb))


# ### Bottom line: Rescaling didn't help us 

# # At last: Creating a simple submission

# In[ ]:


n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))


# In[ ]:


sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    X_time[idx, :] = basic_properties_time(np.abs(train.acoustic_data[samp - samp_length : samp]))
    X_diff[idx, :] = basic_properties_diff(np.diff(train.acoustic_data[samp - samp_length : samp]))
    X_mfcc[idx, :] = mfcc_wrap(train.acoustic_data[samp - samp_length : samp]).ravel()


# # And off to the leader board:

# In[ ]:


model_basic = RandomForestRegressor(n_estimators=100)
model_basic.fit(np.hstack([X_time, X_diff]), y)


# ### We are using very basic features, hoping that the distribution is the same, should not mean too much (I hope)

# In[ ]:


submit_df = test_ids.copy()
for seg_id in test_ids.seg_id:
    temp = pd.read_csv(f'{PATH}test/{seg_id}.csv')
    temp = temp.iloc[::2, :] # we've been using decimated data the whole time...
    x_pred = np.zeros((1, 13))
    x_pred[0, :7] = basic_properties_time(np.abs(temp.acoustic_data))
    x_pred[0, 7:] = basic_properties_diff(np.diff(temp.acoustic_data))
    X_pred = model_basic.predict(x_pred)
    
    submit_df.loc[test_ids.seg_id == seg_id, 'time_to_failure'] = np.max([0, X_pred])


# In[ ]:


submit_df.to_csv('submission.csv', index=False)

