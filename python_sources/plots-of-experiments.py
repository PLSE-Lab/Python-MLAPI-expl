#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_train.info()


# In[ ]:


def figures_subplot(crew, seat):
    print('These plots show the ecg, r, and gsr readings across the three experiments.')
    print('Orange denotes actual event, blue denotes non-event.')
    fig, ax = plt.subplots(3, 3,figsize=(20,20), sharex='col', sharey='row')
    
    for j, y in enumerate(['CA','DA','SS']):
        
        for i, x in enumerate(['ecg','r','gsr']):
            
            
            example = data_train.loc[(data_train.crew==crew)&(data_train.experiment==y)&(data_train.seat==seat)].sort_values(by='time').reset_index()
            ax[i,j].scatter(x=example.loc[example.event=='A'].time, y=example.loc[example.event=='A'][x], s=0.5)
            ax[i,j].set_xlabel('Time')
            ax[i,j].set_ylabel(x)
            ax[i,j].set_title(y)
            if y == 'CA':
                ax[i,j].scatter(x=example.loc[example.event=='C'].time, y=example.loc[example.event=='C'][x], s=0.5)
            elif y == 'DA':
                ax[i,j].scatter(x=example.loc[example.event=='D'].time, y=example.loc[example.event=='D'][x], s=0.5)
            elif y == 'SS':
                ax[i,j].scatter(x=example.loc[example.event=='B'].time, y=example.loc[example.event=='B'][x], s=0.5)
    fig.suptitle('Crew: {}, Seat: {}'.format(crew, seat), fontsize=16)
    plt.show()


# In[ ]:


figures_subplot(1,0)


# In[ ]:


for crew in data_train.crew.unique():
    for seat in range(2):
        figures_subplot(crew, seat)


# A look through these plots show that there are some errors in the experiments readings. Below is a list of errors:
# 
# * Crew: 1, Seat: 0, error in ecg readings for DA.
# * Crew: 2, Seat: 1, error in all gsr readings.
# * Crew: 5, Seat: 1, error in gsr readings for DA.
# * Crew: 7, Seat: 0, error in gsr readings for CA and DA.

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


# In[ ]:


data_test = pd.read_csv('../input/test.csv')


# In[ ]:


def figures_loft(crew, seat):
    print('These plots show the ecg, r, and gsr readings for the test experiment.')
    
    fig, ax = plt.subplots(3, 1, figsize=(20,20), sharex='col', sharey='row')
    
    for i, x in enumerate(['ecg','r','gsr']):
            
        example = data_test.loc[(data_test.crew==crew)&(data_test.seat==seat)].sort_values(by='time').reset_index()
        ax[i].scatter(x=example.time, y=example[x], s=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel(x)
        ax[i].set_title('LOFT')

    fig.suptitle('Crew: {}, Seat: {}'.format(crew, seat), fontsize=16)
    plt.show()


# In[ ]:


figures_loft(1,0)


# In[ ]:


for crew in data_train.crew.unique():
    for seat in range(2):
        figures_loft(crew, seat)


# Looking through the test experiments plot, you can see that there are missing readings for some of the pilots. Below is a list of readings with error:
# 
# * Crew: 1, Seat: 0, errors in gsr readings
# * Crew: 1, Seat: 1, errors in gsr readings
# * Crew: 3, Seat: 0, errors in ecg, gsr readings
# * Crew: 5, Seat: 0 errors in gsr readings
# * Crew: 6, Seat: 1, errors in gsr readings
# * Crew: 7, Seat: 0, errors in gsr readings

# In[ ]:




