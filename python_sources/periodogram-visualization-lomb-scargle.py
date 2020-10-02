#!/usr/bin/env python
# coding: utf-8

# # PLAsTiCC Data Visualization Tools with Periodograms

# ## Intro
# The Lomb-Scargle periodogram (named for Lomb (1976) and Scargle (1982)) is a classic method for finding periodicity in irregularly-sampled data. It is in many ways analogous to the more familiar Fourier Power Spectral Density (PSD) often used for detecting periodicity in regularly-sampled data.
# 
# If we want to classify the sound of two similar objects that emit a certain frequency, let's say C, we would find some problems if the data sources are not perfectly aligned. 
# 
# !['Signal vs Frequency'](https://www.kaggle.com/nanmaran/plasticc/downloads/signal%20vs%20freq.png)
# 
# If we want to get an analysis of the nature of the 2 sounds from the pressure vs time points, at first the 2 datasets seem very different. Where the first one reaches a minimum at t=10, the other one reaches a maximum. But if we observe the periodogram, we can appreciate the similitud between the 2 signals. This is one of the signal interpretation logistics we will use to visualize astronomical sources that vary with time.
# 
# The same way a musical note sounds higher or lower depending on the relative movement of the source and the receptor because of the Doppler effect, astronomical signals will be affected by their distance to the Earth, the farther they are, the faster they are moving away from Earth. So a redshift correction may have to be made to classify the read frequencies of each source.
# 
# 

# ## Load libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import scipy.signal as signal

import timeit as t
import os
print(os.listdir("../input"))
import time
from tqdm import tqdm, tqdm_notebook


# In[ ]:


# Loading the data
df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
df.name = 'Training Set'
df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
df_meta.name = 'Training Metadata Set'


# ## Draw random charts
# This tool was made for having an overview of the different types of astronomical sources. The arguments are:
# *  signalBool: 0 for raw data and 1 for Lomb-Scargle periodogram
# *  f: array of frequencies for Lomb-Scargle analysis

# In[ ]:


def drawRandomCharts(signalBool = 0, f = np.linspace(1, 100, 100)):
    plt.figure(figsize=(20,20))
    for i,target in enumerate(df_meta['target'].unique()):
        rnd = np.random.randint(0, 20)
        object_id = df_meta[df_meta['target'] == target].iloc[rnd].object_id
        plt.subplot(4, 4, i+1)
        for passband in df['passband'].unique():
            passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)]
            plt.title('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                     + '\n Object: ' + str(int(object_id)))
            if (signalBool):
                plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True))
            else:
                plt.plot('mjd', 'flux', data=passbandData )
                plt.scatter('mjd', 'flux', data=passbandData )


# In[ ]:


drawRandomCharts(0)


# In[ ]:


drawRandomCharts(0)


# In[ ]:


drawRandomCharts(1)


# In[ ]:


drawRandomCharts(1, np.linspace(1,10,1000))


# ## Draw passbands charts
# This tool was made for having deeper visualization of an astronomical object, it divides it's values between the different passbands. The arguments are:
# * object_id
# *  signalBool: 0 for raw data and 1 for Lomb-Scargle periodogram
# *  f: array of frequencies for Lomb-Scargle analysis

# In[ ]:


def drawPassbands(object_id, signalBool = 0, f = np.linspace(1, 100, 100)):
    plt.figure(figsize=(18,20))
    for i,passband in enumerate(df['passband'].unique()):
        passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)].copy()
        passbandData['markerSize'] = (passbandData['detected'] + 0.15) * 20
        plt.subplot(6, 1, i+1)
        plt.title('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                 + '\n Object: ' + str(int(object_id)))
        if (signalBool):
            plt.suptitle('\n\nLomb-Scargle frequencies', fontsize=18)
            plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True)) 
        else:
            plt.suptitle('\n\nRaw signals', fontsize=18)
            plt.plot('mjd', 'flux', data=passbandData, alpha=0.3 )
            plt.scatter('mjd', 'flux', 'markerSize' , data=passbandData )


# In[ ]:


drawPassbands(713,0)


# In[ ]:


drawPassbands(713,1)


# ## Astronomical object inspector
# This tool was made for creating a report of a certain astronomical object, including the raw data and the corresponding periodogram. The arguments are:
# * object_id
# *  f: array of frequencies for Lomb-Scargle analysis

# In[ ]:


def starInspector(object_id, f = np.linspace(1, 100, 200)):
    plt.figure(figsize=(18,20), num=1)
    plt.suptitle('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                 + '\n Object: ' + str(int(object_id))
                 + '\n ddf = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'ddf'].iloc[0])
                 + '\n specz = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'hostgal_specz'].iloc[0])
                 + '\n photoz = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'hostgal_photoz'].iloc[0])
                 + '\n distance = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'distmod'].iloc[0])
                 + '\n mwebv = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'mwebv'].iloc[0])
                 + '\n freq = ' + str(f.min()) + ' - ' + str(f.max()) + ' --- ' + str(f.shape)
                 , fontsize=16)
    initTime = t.timeit()
    for i,passband in enumerate(df['passband'].unique()):
        passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)].copy()
        passbandData['markerSize'] = (passbandData['detected'] + 0.15) * 20
        plt.subplot(6, 2, 2*(i+1)-1)
        plt.title('Passband: ' + str(i) + ' (' + str(passbandData.shape[0]) + ') ')
        plt.plot('mjd', 'flux', data=passbandData, alpha=0.3 )
        plt.scatter('mjd', 'flux', 'markerSize' , data=passbandData )

        plt.subplot(6, 2, 2*(i+1))
        plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True))    


# In[ ]:


starInspector(713)


# In[ ]:


starInspector(713, np.linspace(0.1,10,1000))


# Iterating the above function to every object in a target class could give us a good visualization of the behaviour of objects of a certain class.

# In[ ]:




