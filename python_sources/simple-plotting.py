#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### data loadiing

# In[ ]:


data_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
data_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
data_sample_submit = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")


# ### data size

# In[ ]:


print("data_train.shape :", data_train.shape)
print("data_test.shape :", data_test.shape)
print("data_sample_submit.shape :", data_sample_submit.shape)


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_sample_submit.head()


# ### data splitting

# In[ ]:


data_train_dict = {}
data_test_dict = {}

for i in range(data_train.shape[0] // 500000):
    data_train_dict[i] = data_train.iloc[i*500000:i*500000+500000, :]
    
for i in range(data_test.shape[0] // 500000):
    data_test_dict[i] = data_test.iloc[i*500000:i*500000+500000, :]


# ### calculate 1 open channel voltage
# 
# maybe, 1 open channel voltage is not equal to 10 open channel voltage / 10

# In[ ]:


vol = data_train_dict[0]["signal"][data_train_dict[0]["open_channels"]==1].mean() - data_train_dict[0]["signal"][data_train_dict[0]["open_channels"]==0].mean()
print(vol)


# ### plotting

# all data

# In[ ]:


def train_plot(data, plot_range=[0, 500000]):
    """
    data : train data
    plot_range : start point and end point of plot [start point, end point]
    """
    fig, axes = plt.subplots(2, 1, figsize=[12, 8])
    
    axes[0].set_title("data_id_{}_signal".format(i))
    axes[1].set_title("data_id_{}_open_channels".format(i))

    axes[0].plot(data["time"].iloc[plot_range[0]:plot_range[1]], data["signal"].iloc[plot_range[0]:plot_range[1]])
    axes[1].plot(data["time"].iloc[plot_range[0]:plot_range[1]], data["open_channels"].iloc[plot_range[0]:plot_range[1]])
    
    return fig, axes

for i in data_train_dict.keys():
    fig, axes = train_plot(data_train_dict[i])
    
    plt.show()


# enlarged

# In[ ]:


for i in data_train_dict.keys():
    fig, axes = train_plot(data_train_dict[i], [9600, 10000])
    
    plt.show()


# lag

# In[ ]:


def train_lag_plot(data, plot_range=[0, 500000]):
    """
    data : train data
    plot_range : start point and end point of plot [start point, end point]
    """
    fig, axes = plt.subplots(3, 1, figsize=[12, 12])
    
    axes[0].set_title("data_id_{}_signal".format(i))
    axes[1].set_title("data_id_{}_open_channels".format(i))
    axes[2].set_title("data_id_{} signal - open_channels".format(i))


    axes[0].plot(data["time"].iloc[plot_range[0]:plot_range[1]], data["signal"].iloc[plot_range[0]:plot_range[1]])
    axes[1].plot(data["time"].iloc[plot_range[0]:plot_range[1]], data["open_channels"].iloc[plot_range[0]:plot_range[1]])
    axes[2].plot(data["time"].iloc[plot_range[0]:plot_range[1]],
                 data["signal"].iloc[plot_range[0]:plot_range[1]] - vol * data["open_channels"].iloc[plot_range[0]:plot_range[1]])
    
    return fig, axes

for i in data_train_dict.keys():
    fig, axes = train_lag_plot(data_train_dict[i], [9600, 10000])
    
    plt.show()
    
    print()
    print()


# test data

# In[ ]:


def test_plot(data, plot_range=[0, 500000]):
    """
    data : test data
    plot_range : start point and end point of plot [start point, end point]
    """
    fig, axes = plt.subplots(2, 1, figsize=[12, 8])
    
    axes[0].set_title("data_id_{}_signal".format(i))
    axes[1].set_title("data_id_{}_open_channels".format(i))

    axes[0].plot(data["time"].iloc[plot_range[0]:plot_range[1]], data["signal"].iloc[plot_range[0]:plot_range[1]])
    
    return fig, axes

for i in data_test_dict.keys():
    fig, axes = test_plot(data_test_dict[i])
    
    plt.show()

