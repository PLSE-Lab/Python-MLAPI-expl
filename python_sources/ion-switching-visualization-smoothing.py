#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt
import seaborn as sns


# In[ ]:


sample_submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
test_df = pd.read_csv("../input/liverpool-ion-switching/test.csv")
train_df = pd.read_csv("../input/liverpool-ion-switching/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.describe()


# Number of unique values in the column "open_channels"

# In[ ]:


train_df["open_channels"].nunique()


# In[ ]:


test_df.describe()


# Treating unknown values by replacing them with NaN.

# In[ ]:


train_nan = train_df.replace('?', np.nan)
test_nan = test_df.replace('?', np.nan)


# In[ ]:


train_nan.isnull().sum()


# In[ ]:


test_nan.isnull().sum()


# In[ ]:


train_df = train_nan
test_df = test_nan


# ## Time vs. Signal Plot

# In[ ]:


def plot_fig(data, title="Time Series Plot", color = 'b'):
    plt.figure(figsize=(14,7))
    plt.plot(data["time"], data["signal"], color=color)
    plt.title(title, fontsize=28)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Signal", fontsize=20)
    plt.show()


# In[ ]:


plot_fig(data=train_df, title="Train Data", color='b')


# In[ ]:


plot_fig(data=train_df[0:1000], title="Train Data", color='r')


# In[ ]:


plot_fig(test_df, "Test Data", 'g')


# ## Plotting Time vs. Signal and Time vs. Open Channels on a single graph.

# In[ ]:


def plot_fig_combined(data, title="Time Series Plot"):
    plt.figure(figsize=(14,7))
    plt.plot(data["time"], data["signal"], color='r', label='Signal')
    plt.plot(data["time"], data["open_channels"], color='b', label='Open Channels')
    plt.title(title, fontsize=28)
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Signal & Open Channel", fontsize=20)
    plt.show()


# In[ ]:


plot_fig_combined(train_df[0:500], "Signal and Open Channel Variation")


# In[ ]:


plot_fig_combined(train_df[15000:15500], "Signal & Open Channel Variation (1.5s - 1.55s)")


# In[ ]:


plot_fig_combined(train_df[1500000:1500500], "Signal & Open Channel Variation (150s - 150.05s)")


# In[ ]:


plot_fig_combined(train_df[2500000:2500500], "Signal & Open Channel Variation (250s - 250.05s)")


# In[ ]:


plot_fig_combined(train_df[3500000:3500500], "Signal & Open Channel Variation (350s - 350.05s)")


# In[ ]:


plot_fig_combined(train_df[4500000:4500500], "Signal & Open Channel Variation (450s - 450.05s)")


# ## Plotting the Open Channels.

# In[ ]:


def plot_open_channel(data, title):
    plt.figure(figsize=(8,6))
    sns.countplot(data["open_channels"])
    plt.title(title)
    plt.show()


# In[ ]:


plot_open_channel(train_df[:1000000], "Open Channels Variation (0-100s)")


# In[ ]:


plot_open_channel(train_df[1000000:2000000], "Open Channels Variation (100-200s)")


# In[ ]:


plot_open_channel(train_df[2000000:3000000], "Open Channels Variation (200-300s)")


# In[ ]:


plot_open_channel(train_df[3000000:4000000], "Open Channels Variation (300-400s)")


# In[ ]:


plot_open_channel(train_df[4000000:5000000], "Open Channels Variation (400-500s)")


# ## Smoothing of the signal using different window sizes.

# In[ ]:


window_size = [10, 50, 100, 1000]

def comp_plot_signal(data, title):
    for window in window_size:
        smooth_df_mean = train_df.rolling(window).mean()
        smooth_df_sd = train_df.rolling(window).std()
        plt.figure(figsize=(14,7))
        plt.plot(data["time"], data["signal"], color='b', label='Signal')
        plt.plot(data["time"], smooth_df_mean["signal"], color='r', label='Mean of Smoothed Signal with ' + str(window) + ' window size.')
        plt.plot(data["time"], smooth_df_sd["signal"], color='g', label='SD of Smoothed Signal with ' + str(window) + ' window size.')
        plt.title(title)
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Signal", fontsize=20)
        plt.legend(loc='upper right')
        plt.show()   


# In[ ]:


comp_plot_signal(train_df, "Default Signal vs. Smoothed Signal")


# ## Smoothing of the Open Channels using different window sizes.

# In[ ]:


window_size = [10, 50, 100, 1000]

def comp_plot_open_channel(data, title):
    for window in window_size:
        smooth_df_mean = train_df.rolling(window).mean()
        smooth_df_sd = train_df.rolling(window).std()
        plt.figure(figsize=(14,7))
        plt.plot(data["time"], data["open_channels"], color='b', label='Open Channel')
        plt.plot(data["time"], smooth_df_mean["open_channels"], color='r', label='Mean of Smoothed Open Channel with ' + str(window) + ' window size.')
        plt.plot(data["time"], smooth_df_sd["open_channels"], color='g', label='SD of Smoothed Open Channel with ' + str(window) + ' window size.')
        plt.title(title)
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Open Channels", fontsize=20)
        plt.legend(loc='upper right')
        plt.show()


# In[ ]:


comp_plot_open_channel(train_df, "Default Open Channel vs. Smoothed Open Channel")

