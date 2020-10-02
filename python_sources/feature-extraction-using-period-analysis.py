#!/usr/bin/env python
# coding: utf-8

# Light curve can tell you a lot about the type of variable object, especially when the object is periodic. This kernel is about to examine, how to extract additional features from the lightcurves using periodograms and phase curves.
# 
# We will use scipy.signal.lobscargle which is something like fourier analysis for unevenly distributed data. Also known as [Least-squares spectral analysis](https://en.wikipedia.org/wiki/Least-squares_spectral_analysis) (LSSA).

# In[ ]:


import numpy as np
import pandas as pd
import gc
from matplotlib import pyplot as plt
from scipy.signal import lombscargle
import math
from tqdm import tqdm


# In[ ]:


# some help functions
# angular frequency to period
def freq2Period(w):
    return 2 * math.pi / w
# period to angular frequency
def period2Freq(T):
    return 2 * math.pi / T


# In[ ]:


gc.enable()
train = pd.read_csv('../input/training_set.csv')
print(train['object_id'].unique())
gc.collect()


# My basic idea is simple: take lightcurves in different bands and normalize them, so that they fit together. Most periodic changes happen in all bands. The difference is, bands are usually shifted from each other (they have different **mean**), and the amplitude of changes can be also different (they have different **standard deviation**). Normalizing bands will make most of the well-behaved variable object fit together.
# 
# *Note: this is just some rough approximation, which may not work for exotic (mainly extra-galactic) or extreme objects (like active black holes or whatever) - in general any object, whose light curves are not similar enough in different bands. But the information about such a mismatch is valuable for classifications by itself. For such objects, you could do periodogram for each band and then merge the results. But not know, maybe in future versions...*

# In[ ]:


# get data and normalize bands
def processData(train, object_id):
    
    #load data for given object
    X = train.loc[train['object_id'] == object_id]
    x = np.array(X['mjd'].values)
    y = np.array(X['flux'].values)
    passband = np.array(X['passband'].values)
    
    # normalize bands
    for i in np.unique(passband):
        yy = y[np.where(passband==i)]
        mean = np.mean(yy)
        std = np.std(yy)
        y[np.where(passband==i)] = (yy - mean)/std
    
    return x, y, passband


# Let's get light curve for first object in dataset and plot it.

# In[ ]:


x, y, passband = processData(train, 615)
plt.scatter(x, y, c=passband)
plt.xlabel('time (MJD)')
plt.ylabel('Normalized flux')
plt.show()


# Big mess, right? Let's make some sense in it by periodograms.
# Loosely speaking, periodogram shows you something like the probability for each of possible periods (well, not exactly probability, that's why I call it power, but it's enough for basic understanding).
# 
# The computation takes ages and time is our most valuable resource (we have more than 3M objects in test set). Therefore I look for 5 most "probable" periods above some "probability" threshold and use them as new features.

# In[ ]:


# calculate periodogram
def getPeriodogram(x, y, steps = 10000, minPeriod = None, maxPeriod = None):
    if not minPeriod:
        minPeriod = 0.1 # for now, let's ignore very short periodic objects
    if not maxPeriod:
        maxPeriod = (np.max(x) - np.min(x))/2 # you cannot detect P > half of your observation period

    maxFreq = np.log2(period2Freq(minPeriod))
    minFreq = np.log2(period2Freq(maxPeriod))
    f = np.power(2, np.linspace(minFreq,maxFreq, steps))
    p = lombscargle(x,y,f,normalize=True)
    return f, p


# In[ ]:


get_ipython().run_cell_magic('time', '', 'f,p = getPeriodogram(x, y, steps=20000)')


# In[ ]:


plt.semilogx(freq2Period(f),p)
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.show()


# You can see how a typical periodogram of noisy data looks like. There is huge noise and several peak of different size. We use only 10000 steps for first pass, which means we may not always hit the peak exactly. Let's take all peak candidates of certain height (let's say power > 0.3) and examine them further.

# In[ ]:


def findBestPeaks(x, y, F, P, threshold=0.3, n=5):
    
    # find peaks above threshold
    indexes = np.where(P>threshold)[0]
    # if nothing found, look at the highest peaks anyway
    if len(indexes) == 0:
        q = np.quantile(P, 0.9995)
        indexes = np.where(P>q)[0]
    
    peaks = []
    start = 0
    end = 0
    for i in indexes:
        if i - end > 10:
            peaks.append((start, end))
            start = i
            end = i
        else:
            end = i
    
    peaks.append((start, end))
        
    
    # increase accuracy on the found peaks
    results = []
    for start, end in peaks:
        if end > 0:
            minPeriod = freq2Period(F[min(F.shape[0]-1, end+1)])
            maxPeriod = freq2Period(F[max(start-1, 0)])
            steps = int(100 * np.sqrt(end-start+1)) # the bigger the peak width, the more steps we want - but sensible (linear increase leads to long computation)
            f, p = getPeriodogram(x, y, steps = steps, minPeriod=minPeriod, maxPeriod=maxPeriod)
            results.append(np.array([freq2Period(f[np.argmax(p)]), np.max(p)]))

    # sort by normalized periodogram score and return first n results
    if results:
        results = np.array(results)
        results = results[np.flip(results[:,1].argsort())]
    else:
        results = np.array([freq2Period(F[np.argmax(P)]), np.max(P)]).reshape(1,2)
    return results[0:n]


# In[ ]:


get_ipython().run_cell_magic('time', '', "results = findBestPeaks(x, y, f, p)\nprint('Period(days) Power')\nprint(results)")


# We found 4 peaks - one with a very high power, three others with much lower one. Let's check the results visually:

# In[ ]:


plt.figure(figsize=(20,25))

for i in range(results.shape[0]):
    plt.subplot(results.shape[0],2,i+1)
    phase = x/results[i][0] % 1
    plt.scatter(phase, y, c = passband, s=4)
    plt.xlabel('Phase')
    plt.ylabel('Normalized flux')
    plt.title('Period: {:.4f}, power: {:.2f}'.format(results[i][0], results[i][1]))

plt.show()


# We can see that only first period makes sense and the rest are false positives. This one looks like some short-period variable star with nicely periodic changes.
# 
# You can use the found periods and their respective powers as a new feature.
# 
# With phase curve, you can also try to extract other features, e.g.:
# * shape
# * symmetry / assymetry of the curve
# * humps, double minimas / maximas
# * fit the phase curve with sin function and calculate residuals for each band - in combination with flux errors, it's a measure of how strong the periodicity is (some objects are nicely periodic, like this one, some are semi-periodic with each minimum/maximum slightly different, which makes the phase curve more noisy)

# # Multiprocessing
# 
# In this section, I will try to develop speed optimized technique to precompute basic periodogram features.

# In[ ]:


from multiprocessing import Pool
import multiprocessing as mp

CORES = mp.cpu_count() #4

def getFeatures(object_id):
    
    x, y, passband = processData(train, object_id)
    f,p = getPeriodogram(x, y)
    peaks = findBestPeaks(x, y, f, p)
    features = np.zeros((5,2))
    features[:peaks.shape[0],:peaks.shape[1]] = peaks
    
    return np.append(np.array([object_id]), features.reshape(5*2))


# In[ ]:


object_ids = train['object_id'].unique()[0:100]


# First, let's try to calculate without multi-cpu speed-up.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'features = []\nfor object_id in object_ids:\n    results = getFeatures(object_id)\n    features.append(results)')


# That's 0.284 second per star. Training set will then take 2200 seconds to calculate. Testing set would take 852000 seconds, ~10 days. Not good.
# Let's try 4 cores available in Kaggle Kernels.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'p = Pool(CORES)\n\nresults = p.map(getFeatures, object_ids)')


# That's 0.100 seconds per star. Training set will take around 780 seconds. Testing set 300 000 seconds (~3,5 days). With 4x more CPU power, we got roughly 2,8x speedup.
# If this relation holds linearly with CPU power, we could get testing set computed in 10 hours on 32 CPU on google cloud.
# I will definitelly try and will make it public, if the new features prove to be benefitial on train/validation set.

# # Calculating training set

# In[ ]:


object_ids = train['object_id'].unique()
columns = np.array(['id'])
for i in range(5):
    period_str = 'period_'+str(i+1)
    power_str = 'power_'+str(i+1)
    columns = np.append(columns, np.array([period_str, power_str]))

results = p.map(getFeatures, object_ids)

output = pd.DataFrame(results, columns=columns)
output['id'] = output['id'].astype(np.int32)
output.to_csv('./train-periods.csv')


# # Further development / ideas:
# 
# * *the speed is a key.  The process to compute periodograms and features for 3M+ dataset cannot take ages. I.e. we need paralelization and smart optimization of the number steps in periodogram search*
# * sort the observations by phase and band and feed the phase curve into RNN
# * feed the phase curve into CNN with channels = number of bands
# * look for the functions, that fit the curves well - some classes of variable objects can be fitted very precisely by a specific function, which can then help to identify 99 class (increase your chance to have it right).
