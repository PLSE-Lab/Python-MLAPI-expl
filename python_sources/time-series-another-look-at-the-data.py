#!/usr/bin/env python
# coding: utf-8

# In most of the public kernels of this competition, people are using ensemble techniques to predict the different states of the pilot at each time step. However, the input data can be seen as time series and related tools can be used.
# 
# So the aim of this kernel is to use  time series classifiers on the input data. The main problem with this task is that there's only the features variations of 18 pilots over time, so we can't just feed them directly to those models.
# 
# I borrowed some of the preprocessing functions / visualizations to those great kernels, that I recomment you to read first:
# * https://www.kaggle.com/theoviel/starter-code-eda-and-lgbm-baseline
# * https://www.kaggle.com/ashishpatel26/smote-with-model-lightgbm

# In[ ]:


import warnings
import itertools
import numpy as np 
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
import dask.dataframe as dd
import dask
import gc
import matplotlib.pyplot as plt
import itertools

from yellowbrick.text import TSNEVisualizer

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")

import os
print(os.listdir("../input"))

warnings.simplefilter(action='ignore')
sns.set_style('whitegrid')


# In[ ]:


dtypes = {"crew": "int8",
          "experiment": "category",
          "time": "float32",
          "seat": "int8",
          "eeg_fp1": "float32",
          "eeg_f7": "float32",
          "eeg_f8": "float32",
          "eeg_t4": "float32",
          "eeg_t6": "float32",
          "eeg_t5": "float32",
          "eeg_t3": "float32",
          "eeg_fp2": "float32",
          "eeg_o1": "float32",
          "eeg_p3": "float32",
          "eeg_pz": "float32",
          "eeg_f3": "float32",
          "eeg_fz": "float32",
          "eeg_f4": "float32",
          "eeg_c4": "float32",
          "eeg_p4": "float32",
          "eeg_poz": "float32",
          "eeg_c3": "float32",
          "eeg_cz": "float32",
          "eeg_o2": "float32",
          "ecg": "float32",
          "r": "float32",
          "gsr": "float32",
          "event": "category",
         }

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Load Data

# In[ ]:


train = dd.read_csv("../input/train.csv", blocksize= 256e6, dtype = dtypes)
test = dd.read_csv("../input/test.csv", blocksize= 256e6, dtype = dtypes)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train  = train.compute()\nprint("Training shape : ", train.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test  = test.compute()\nprint("Training shape : ", test.shape)')


# # Processing data

# In[ ]:


dic_exp = {'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': -1}
# A = baseline, B = SS, C = CA, D = DA
dic_event = {'A': -1, 'B': 2, 'C': 0, 'D': 1}

labels_exp = {v: k for k, v in dic_exp.items()}
labels_event = {v: k for k, v in dic_event.items()}

train["event"] = train["event"].apply(lambda x: dic_event[x])
train["event"] = train["event"].astype('int8')
train['experiment'] = train['experiment'].apply(lambda x: dic_exp[x])
test['experiment'] = test['experiment'].apply(lambda x: dic_exp[x])

train['experiment'] = train['experiment'].astype('int8')
test['experiment'] = test['experiment'].astype('int8')


# As mentionned in the data description of the competition, each of the 18 pilots was recorded over time and subjected to the CA, DA, or SS cognitive states. The training set contains three experiments (one for each state) in which the pilots experienced just one of the states.
# 
# Thus, we have to hierarchically group the training data by :
# * crew
# * seat (pilot)
# * experiment

# In[ ]:


train = train.set_index(['crew', 'seat', 'experiment'])


# In[ ]:


train.head()


# Let's look at the data about a single pilot and visualize the evolution of different features over time.

# In[ ]:


# Pilot in the seat 0 of the crew 1
pilot = train.loc[1, 0]


# If we plot the time evolution through index, we see that the values are not sorted. Let's sort them (and do it for all the data).

# In[ ]:


plt.figure(figsize=(7,5))
plt.plot(range(len(pilot.loc[0])), pilot.loc[0].time)
plt.xlabel('index')
plt.ylabel('time')
plt.show()


# In[ ]:


# Sort the values by increasing time
train = train.sort_values(by='time').sort_index()
pilot = train.loc[1, 0]


# In[ ]:


# areas colors corresponding to the event state of the pilot
# baseline: gray, CA: green, SS: red, DA: blue
event_colors = {0: 'green', 1: 'blue', 2: 'red', -1: 'gray'}

def plot_ts(pilot_data, features, exp=0):
    exp_data = pilot_data.loc[exp]
    
    ax = exp_data.plot(
         kind='line',
         x='time', 
         y=features, 
         figsize=(15,5), 
         linewidth=2.
    )
    changes = exp_data[exp_data.event.diff().abs()>0][['time', 'event']].values
    times = [0] + list(changes[:, 0]) + [exp_data.time.max()]
    events = [exp_data.event.iloc[0]] + list(changes[:, 1])
    for i in range(len(times)-1):
        event = events[i]
        ax.axvspan(times[i], times[i+1], facecolor=event_colors[event], alpha=0.1)
    
    plt.show()


# Here is the time evolution of the feature `event` over time, which are also represented by the color backgrounds, so we can assure that the plotting function is correct.

# In[ ]:


plot_ts(pilot_data=pilot, features='event', exp=0)
plot_ts(pilot_data=pilot, features='event', exp=1)
plot_ts(pilot_data=pilot, features='event', exp=2)


# Here is the evolution of several interesting features :

# In[ ]:


plot_ts(pilot_data=pilot, features='gsr', exp=0)
plot_ts(pilot_data=pilot, features='gsr', exp=1)
plot_ts(pilot_data=pilot, features='gsr', exp=2)


# In[ ]:


# Most are very noisy
f = ['eeg_f4', 'eeg_c4', 'eeg_p4']
plot_ts(pilot_data=pilot, features=f, exp=0)
plot_ts(pilot_data=pilot, features=f, exp=1)
plot_ts(pilot_data=pilot, features=f, exp=2)


# Here is the plots for another pilot

# In[ ]:


pilot = train.loc[5, 0]
f = ['gsr']
plot_ts(pilot_data=pilot, features=f, exp=0)
plot_ts(pilot_data=pilot, features=f, exp=1)
plot_ts(pilot_data=pilot, features=f, exp=2)


# In[ ]:


f = ['r']
plot_ts(pilot_data=pilot, features=f, exp=0)
plot_ts(pilot_data=pilot, features=f, exp=1)
plot_ts(pilot_data=pilot, features=f, exp=2)


# We can notice that each experiment has always the same time shape :
# * `CA` experiment begins with a small time of baseline and continues with `CA`state, 
# * `DA` is baseline states in which some some little periods of `DA`state appear, 
# * `SA` experiment is baseline state in which one or more `SA` states (jump scares) appear.
# 
# Unfortunately, the test data does not provide us the experiment labels (only `LOFT`), otherwise it would have been easy... 

# ** More to come !**
# 
# ** TODO :**
# * Find a way to have more (smaller) samples to train classifiers
# * Test some time series tools (RNN, Convolutions, ...)
# 
# **Thanks for reading, and I'd be happy to have your feedback! **
