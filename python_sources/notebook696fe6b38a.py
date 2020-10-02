#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use('ggplot')

import seaborn as sns; sns.set(style="ticks", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[39]:


df = pd.read_csv('../input/SLC_Police_Cases_2016_cleaned_geocoded.csv')


# In[40]:


|df['reported'] = pd.to_datetime(df['reported'], errors='coerce')
df['occurred'] = pd.to_datetime(df['occurred'], errors='coerce')


# In[41]:


df["occur_date"] = df["occurred"].dt.date
df["occur_month"] = df["occurred"].dt.month
df["occur_hour"] = df["occurred"].dt.hour
df["report_date"] = df["reported"].dt.date
df["report_month"] = df["reported"].dt.month
df["report_hour"] = df["reported"].dt.hour
df.head()


# ### Top Crimes
# 

# In[44]:


newdf = df.groupby(["description","occur_date"]).size().reset_index(name="Count")


# In[47]:


g = sns.FacetGrid(newdf,col="description",col_wrap=7)
g = g.map(sns.pointplot,"occur_date","Count")

