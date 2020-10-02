#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Euroleague dataset, to check games' totals.

# In[ ]:


erl = pd.read_csv('/kaggle/input/euroleague-basketball-results-20032019/euroleague_dset_csv_03_20_upd.csv')


# Converting DATE column to needed format 

# In[ ]:


erl['DATE'] = pd.to_datetime(erl['DATE'], format='%d/%m/%Y',infer_datetime_format=True)


# Checking head of the data

# In[ ]:


erl.head()


# Describe the table. We can see here some interesting data.

# In[ ]:


erl.describe()


# Changing types of Home and away team cell types

# In[ ]:


erl['HT'] = erl['HT'].astype(str)
erl['AT'] = erl['AT'].astype(str)


# Adding new columns with just months and years, needed for analysis

# In[ ]:


erl["Month"] = erl["DATE"].dt.strftime('%m')
erl["Year"] = erl["DATE"].dt.strftime('%Y')


# Creating DataFrame "scores" with Full Time Totals.

# In[ ]:


scores = pd.DataFrame(erl.groupby(["FTT"]).FTT.count().sort_values(ascending=False))


# In[ ]:


scores.rename(columns={'FTT':'COUNT'}, inplace=True)


# In[ ]:


scores.reset_index(inplace=True)


# In[ ]:


scores


# Barplot for totals distribution

# In[ ]:


plt.figure(figsize=(50, 10))
sns.barplot(x="FTT", y="COUNT", data=scores)


# Here we can grab statistics only per team name - see operand | (OR) - team can't play home and away same time.

# In[ ]:


erl[erl.HT.str.contains('Barcelona') | erl.AT.str.contains('Barcelona')]


# Another selection with away team and year (season) criteria.

# In[ ]:


pan = erl.loc[(erl.HT=='Panathinaikos') & (erl.Year=="2018")]


# 1. That's how Panathinaicos played home games in 2018

# In[ ]:


pan.describe()


# Let's check created dataframe for Panathinaicos in 2018

# In[ ]:


pan


# Let's check if there was series (like x straight wins). We will make it by cimparing previous row (by shift).

# Let's make DataFrame and reset index (for easy further actionsm with analysis).

# In[ ]:


series_check = pd.DataFrame(pan)


# In[ ]:


series_check.reset_index(inplace=True)


# In[ ]:


series_check


# Let's check if these are series.

# In[ ]:


series_check.loc[series_check["WINNER"] == series_check["WINNER"].shift(1),'RESULT'] = 'WIN'
series_check.loc[series_check["WINNER"] != series_check["WINNER"].shift(1),'RESULT'] = 'LOSE'
series_check.loc[series_check["HT"] == series_check["WINNER"],'RESULT'] = 'SERIE'


# In[ ]:


series_check


# In[ ]:


sns.lmplot(x="FTT",y="GAPT",hue="RESULT", data=series_check)


# In[ ]:


sns.barplot(x="RESULT",y="GAPT",data=series_check)


# Another analysis type I wanted to check - difference between first half and second half, for each dataset game.

# Let's assume 1st half = 1.

# In[ ]:


a = pd.DataFrame(erl["P1T"]/erl["P1T"])
a.reset_index(inplace=True)


# Let's divide first second half to first half, and add it to another column in dataframe.

# In[ ]:


b = pd.DataFrame(erl["P2T"]/erl["P1T"])
b.reset_index(inplace=True)


# Merging.

# In[ ]:


c = pd.merge(a,b)


# In[ ]:


c.rename(columns={0:'P2_cf'}, inplace=True)


# In[ ]:


list(c.columns)


# In[ ]:


c


# In[ ]:


c.groupby([c.P2_cf>1]).index.count()


# Let's see the distribution, per game.

# In[ ]:


c.plot.line(x="index",y="P2_cf", figsize=(200,20))


# And let's count AWAY TEAM totals per 1st quarter, per each game, and vizualize distribution.

# In[ ]:


x = pd.DataFrame(erl.groupby("Q1A").Q1A.count())


# In[ ]:


x

