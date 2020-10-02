#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# # 1. Loading dataset.
# **Data loading with official statistics about spreading coronavirus COVID-19.**

# In[ ]:


df_train = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df_train = df_train.drop(labels=['SNo','Last Update'], axis=1)
df_train['ObservationDate']= pd.to_datetime(df_train['ObservationDate'], format='%m/%d/%Y')
df_train


# In[ ]:


df_train.info()


# In[ ]:


#function to print observation period
def print_obs_period(x):
    min_x = min(x['ObservationDate'])
    max_x = max(x['ObservationDate'])
    print("Observation period: " + str(min_x) + " .. " + str(max_x))


# In[ ]:


#function to plot graph
def plot_graph(x, legend_title):
    fig = plt.figure(figsize=(12,6))
    plt.xticks(rotation=90)
    plt.xscale('linear')
    plt.xlabel("Observations")
    plt.ylabel("People")
    plt.scatter(x['ObservationDate'], x['Confirmed'], c='#1f77b4', label="Confirmed");
    plt.scatter(x['ObservationDate'], x['Deaths'],    c='#ff7f0e', label="Deaths");
    plt.scatter(x['ObservationDate'], x['Recovered'], c='#2ca02c', label="Recovered");
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.show()


# # 2. Provinces of China.
# **The most provinces is reached the final stage of the spread of coronavirus COVID-19**

# In[ ]:


train_cn = df_train[df_train["Country/Region"] == "Mainland China"]
train_cn = train_cn.groupby(["Country/Region","ObservationDate","Province/State"], as_index=False).max()
#train_cn.info()

print_obs_period(train_cn)

train_max = train_cn.groupby(["Province/State"], as_index=False).max()
#train_cn_max.info()

for province in train_max[train_max['Confirmed'] >= 1000]['Province/State']:
    #print(province)
    plot_graph(train_cn[train_cn['Province/State'] == province], province)


# # 3. The world countries excluding China.

# In[ ]:


train_ca = df_train[df_train["Country/Region"] == "Canada"]
train_ca = train_ca.groupby(["Country/Region","ObservationDate","Province/State"], as_index=False).max()                    .groupby(["Country/Region","ObservationDate"], as_index=False).sum()
#train_ca.info()

train_us = df_train[df_train["Country/Region"] == "US"]
train_us = train_us.groupby(["Country/Region","ObservationDate","Province/State"], as_index=False).max()                    .groupby(["Country/Region","ObservationDate"], as_index=False).sum()
#train_us.info()

train_rest = df_train[df_train["Country/Region"] != "Mainland China"][pd.isna(df_train["Province/State"])]
train_rest = train_rest.groupby(["Country/Region","ObservationDate"], as_index=False).max()                        .drop(labels=["Province/State"], axis=1)
#train_rest.info()

train_world = pd.concat([train_cn.groupby(["Country/Region","ObservationDate"], as_index=False).sum(),                          train_ca,
                         train_us,
                         train_rest])
#train_world.info()

print_obs_period(train_world)

train_max = train_world.groupby(["Country/Region"], as_index=False).max()
#train_max.info()

for country in train_max[train_max['Confirmed'] >= 1000]['Country/Region']:
    #print(country)
    plot_graph(train_world[train_world['Country/Region'] == country], country)


# # 4. The whole world.

# In[ ]:


plot_graph(train_world.groupby(["ObservationDate"], as_index=False).sum(), 'World')

