#!/usr/bin/env python
# coding: utf-8

# Learning Jupyter in a real world context

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


# Libraries for constructing graphs

import matplotlib.pyplot as plt
import seaborn as sns


# Load temperature data for Belem

# In[ ]:


df_temperature_records_belem = pd.read_csv("../input/temperature-timeseries-for-some-brazilian-cities/station_belem.csv",)

df_temperature_records_belem.head(68)


# Use Year for index of DataFrame intend to examine sequential decades

# In[ ]:


df_temperature_records_belem.set_index("YEAR", inplace=True)

df_temperature_records_belem


# In[ ]:


# replace missing values (apparently 999.90) with numpy NaN

df_temperature_records_belem.replace(999.90, np.nan, inplace=True)

df_temperature_records_belem


# Want to compare records by decade to see if there is any significant variation up or down between mean temperatures for the decades 1960-9, 1970-9, 1980-9, 1990-9, 2000-9 and 2010-9.
# 1. Split dataset into decades by rows and monthly, seasonal and annual by columns
#     a. Rows first then columns
# 2. Calculate mean temperature for month with standard deviation
# 3. Plot mean with standard deviation error bars for the 6 available decades

# In[ ]:


# define start and finish years and start finish columns for the desired
# dataframes

df_60monthly_belem = df_temperature_records_belem.loc["1961":"1969", "JAN":"DEC"]

df_60monthly_belem


# In[ ]:


df_60monthly_mean_belem = df_60monthly_belem.mean(axis=0)
df_60monthly_mean_belem

