#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/commodity_trade_statistics_data.csv", na_values=["No Quantity",0.0,''],sep=',')


# In[ ]:


df.shape


# In[ ]:


df.year.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df["year"] = df["year"]+1  # we want the date to be the first day of the next year


# In[ ]:


df["year"] = pd.to_datetime(df["year"],format="%Y")


# In[ ]:


df.head()


# In[ ]:


df.drop(["weight_kg","quantity_name"],axis=1,inplace=True)


# In[ ]:


df = df.dropna(how='any').reset_index(drop=True)
df.shape


# In[ ]:


df.flow.value_counts()


# In[ ]:


df[df.flow=="Re-Export"].head()


# In[ ]:


df.to_csv("commodity_trade_stats_global.csv.gz",index=False,compression="gzip")

