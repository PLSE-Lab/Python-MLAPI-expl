#!/usr/bin/env python
# coding: utf-8

# **Introduction: Initial analysis of training data **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# As we know agenda of this challenge is to investigate how coronavirus pandemic has changed oil markets. 
# To solve this problem, we have below data files in hand.
# * Crude_oil_trend_dataset_From1986-10-16_To2020-03-31 - the training set for the time-series of the objective variable.
# * NTL-dataset.zip - the NTL images (tif files) with its numpy files. The numpy files contain missing values as np.nan.
# * COVID-19_train.csv - the training set that is included COVID-19 data and the objective variable.
# * COVID-19_test.csv - the test set.
# * sampleSubmission.csv - a sample submission file in the correct format.

# In this notebook we will just explore data available in COVID-19_train.csv

# In[ ]:


#Import libs
import matplotlib.pyplot as plt


# In[ ]:


#load train.csv data
train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv')


# In[ ]:


train.shape


# so train dataset have just 63 rows/records and quite high number of colum/features(i.e 846).
# We will now have a look on features availabe.... 

# In[ ]:


train.head(10)


# * First column represent to Date and last column represnt to oil price on that perticular date.
# * along with these two column, we have 844(846 - 2) column in between.
# * these column represents region wise COVID situation.
# * for each region there are 4 colums, which represnets : total_case, news_cases,total_deaths,new_deaths
# * for example for region Aruba we have 4 coulms: Aruba_total_cases,Aruba_new_cases,Aruba_total_deaths,Aruba_new_deaths
# * We can also see,records for few dates are missing.ex- Data records for 2020-01-04 and 2020-01-05 are not there,reason behind these days were either weekends or Holidays

# In this file we have data availabe for from date 31-12-2019 to 31-03-2020(3 Month). We look how oil price changed during this period.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['Price'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="Oil Price",
       title="Oil Price: 31-12-2019 to 31-03-2020 ")


# We can now look how COVID case changed during this period. In this file we have data available for each region, but we will just have look for last 4 column which represents world's total number of cases.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_total_cases'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_total_cases",
       title="World_total_cases: 31-12-2019 to 31-03-2020 ")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_new_cases'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_new_cases",
       title="World_new_cases: 31-12-2019 to 31-03-2020 ")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_total_deaths'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_total_deaths",
       title="World_total_deaths: 31-12-2019 to 31-03-2020 ")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_new_deaths'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_new_deaths",
       title="World_new_deaths: 31-12-2019 to 31-03-2020 ")

