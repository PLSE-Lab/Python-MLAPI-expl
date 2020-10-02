#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Exercise 1 from Introduction to SAS (adapted to pandas)

# In[ ]:


df = pd.read_csv("../input/diamonds.csv")
df = df.drop(["Unnamed: 0"],axis=1)
print(df.head())
print(df.info())


# In[ ]:


df_stupid = df["depth"]
print(df_stupid.head())


# In[ ]:


df_SI = df[(df["clarity"].str[0:2] == "SI") & (df["price"] < 400)]
print(df_SI.head())


# In[ ]:


df_calcDepth = df
df_calcDepth["calcDepth"] = 2*df["z"] / (df["x"] + df["y"])
df_calcDepth_filt = df_calcDepth[(df["calcDepth"]<60) & (df["clarity"].str[:2] == "VS")]
print(df_calcDepth_filt.head())

print(df.count())
print(df_calcDepth.count())
print(df_calcDepth_filt.count())

print(df["cut"].value_counts())


# In[ ]:


df_calcDepth_dropped = df_calcDepth.drop("depth",axis=1)
print(df_calcDepth_dropped.head())
print(df_calcDepth.head())


# In[ ]:


df_calcDepth_dropped["calcDepth"] = df_calcDepth_dropped["calcDepth"].round(2)
print(df_calcDepth_dropped.head())


# In[ ]:


sns.


# In[ ]:




