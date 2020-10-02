#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
path =''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path =os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv(path)


# In[ ]:


df.head(5)


# In[ ]:


mean_df = df.mean()
mean_df


# In[ ]:


def plot_func(mean_df,title):
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(mean_df, autopct=lambda pct: func(pct),
                                      textprops=dict(color="w"))

    ax.legend(wedges, ["Delhi","Kolkata","Mumbai","Chennai"],
              title=title,
              loc="center",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=20, weight="bold")

    ax.set_title(title)
    plt.show()


# In[ ]:


plot_func(mean_df ,"Distribution of Mean")


# # Inference
# 
#  Mumbai has the highest average price  and Delhi the Lowest

# # Year Wise Analysis

# In[ ]:


def return_corresponding_year_df(year):
    return df[df["Date"].str.contains(year)]


# ## 2014

# In[ ]:


df_2014 = return_corresponding_year_df('2014')


# In[ ]:


df_2014


# In[ ]:


mean_2014 = df_2014.mean()


# In[ ]:


mean_2014


# In[ ]:


plot_func(mean_2014,"Distribution of Mean 2014")


# In[ ]:


df_2014.describe()


# # Inference of 2014
# 
#  - Mumbai has the highest mean Price .
#  - Mumbai had the highest price

# # Inference of 2015

# In[ ]:


df_2015 = return_corresponding_year_df('2015')


# In[ ]:


df_2015


# In[ ]:


mean_2015 = df_2015.mean()


# In[ ]:


plot_func(mean_2015,"Distibution of Mean 2015")


# In[ ]:


df_2015.describe()


# # Inference 
# 
#   - Mumbai has the highest price 
#   -  Delhii has the lowest price 

# In[ ]:




