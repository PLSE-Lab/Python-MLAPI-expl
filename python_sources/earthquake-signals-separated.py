#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# With this kernel, I separate the entire training file into 16 distinct signals which I will process in different kernels.

# In[2]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
pd.options.display.precision = 12
skipped_rows = 0


# In[3]:


def plot_signal(df):
    fig, ax = plt.subplots(2,1, figsize=(20,12))
    ax[0].plot(df.index.values, df.quaketime.values, c="darkred")
    ax[0].set_title("Quaketime")
    ax[0].set_xlabel("Index")
    ax[0].set_ylabel("Quaketime in ms");
    ax[0].grid(axis = 'y')
    ax[1].plot(df.index.values, df.signal.values, c="mediumseagreen")
    ax[1].set_title("Signal")
    ax[1].set_xlabel("Index")
    ax[1].set_ylabel("Acoustic Signal");
    ax[1].grid(axis = 'y')
    plt.show()


# ## **Signal 01**

# In[4]:


train = pd.read_csv('../input/train.csv', nrows=20000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)


# In[5]:


plot_signal(train)


# In[6]:


#train.head(10)
x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[7]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal01.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# So the cutoff point is `index = 5,656,573`. Then the next set of values will start at `index = 5656574`. We will now find out where's the next cutoff point.

# ## **Signal 02**

# In[8]:


train = pd.read_csv('../input/train.csv', nrows=45000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)


# In[9]:


plot_signal(train)


# In[10]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[11]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal02.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 44,429,303`. Next signal starts at file line 5,656,575 + 44,429,304 = 50,085,879

# ## **Signal 03**

# In[12]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal03.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# So cutoff point is at `index = 54,591,477`. Therefore, the next signal will start at file line = 50,085,879 + 54,591,477 = 104,677,356 + 1 because headers.
# 

# ## **Signal 04**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = 104677357, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .002 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal04.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 34,095,096` for this signal. Next signal will start at file line 104,677,357 + 34,095,097 = 138,772,454

# ## **Signal 05**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = 138772454, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)


# In[ ]:


train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal05.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 48,869,366` for this signal. Next signal will start at file line 138,772,454 + 48,869,366 = 187,641,820 + 1.

# ## **Signal 06**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=32000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .002 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal06.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 31,010,809` for this signal. Next signal will start at file line 187,641,821 + 31,010,810.

# ## **Signal 07**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal07.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 27,176,954`. Next signal will start at 218,652,631 + 27,176,955 = 245,829,586 

# ## **Signal 08**

# In[ ]:


train = pd.read_csv('../input/train.csv'', nrows=65000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal08.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 62,009,331`. Next signal will start at 245,829,586 + 62,009,332 = 307,838,918 

# ## **Signal 09**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=31000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal09.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# Cutoff point is `index = 30,437,369`. Next signal will start at 307,838,918 + 30,437,370 = 338,276,288

# ## **Signal 10**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=38000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .0004956 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal10.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 11**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=45000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .0004956 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal11.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print(f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# In[ ]:


print('cutoff_point = ',f'{cutoff_point:10,d}')


# ## **Signal 12**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=45000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal12.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 13**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=45000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal13.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 14**

# In[ ]:


skipped_rows = 495800226
train = pd.read_csv('../input/train.csv', nrows=40000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal14.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 15**

# In[ ]:


skipped_rows = 528777116
train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal15.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 16**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < .001 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal16.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')


# ## **Signal 17 (Last signal)**

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=60000000, skiprows = skipped_rows, names = ['acoustic_data', 'time_to_failure'],
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head()


# In[ ]:


plot_signal(train)


# In[ ]:


x = train[train['quaketime'] < 9.8 ]
print(x.shape)
x.tail()


# In[ ]:


cutoff_point = x.tail().index[4]
print(train.loc[(train.index >= x.tail().index[0]) & (train.index <= cutoff_point+1)])
train.loc[train.index <= cutoff_point].to_csv('Signal17.csv', float_format='%15.10f')
skipped_rows = skipped_rows + cutoff_point + 1
print('cutoff_point = ',f'{cutoff_point:10,d}')
print('Skipped Rows = ',f'{skipped_rows:10,d}' + ' -- ' + f'{skipped_rows:10d}')

