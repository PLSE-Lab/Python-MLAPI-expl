#!/usr/bin/env python
# coding: utf-8

# **Exploring Your Dataset**

# Figuring out where the Kaggle Dataset is. Default code provided by Kaggle below:

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


get_ipython().system('pwd')
get_ipython().system('ls')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt


# Great, we've loaded in the basic libraries, let's get out Target and start the modelling process.

# In[ ]:


df=pd.read_csv('/kaggle/input/peace-agreements-dataset/pax_20_02_2018_1_CSV.csv')
print(df.shape)
print('')
print(df.dtypes.value_counts())
df.sample(30)


# At first glance, we can assume we're dealing with some categorical fields. Countries, continents, territories. And some kind of 'Status' 

# The last column is probably our Target. Let's check how balanced it is.a

# In[ ]:


df.ImSrc.value_counts()


# 1518? This means ALL values are 1 for our presumed Target column. What's going on with this dataset? Let's explore.

# Can we deduce anything from the column names?

# In[ ]:


df.columns.tolist() # using tolist to display all columns


# There are 240 columns. Way too many to manually skim through. Let's try something else.

# In[ ]:


df.iloc[:,0:20].sample()
# df[['Stage', 'StageSub', 'Part', 'ThrdPart', 'OthAgr', 'GCh', 'GChRhet', 'GChAntid', 'GChSubs', 'GChOth', 'GDis']]


# In[ ]:


df.info()


# In[ ]:


df.Con.unique()


# In[ ]:


df.Con.value_counts() #158 different countries


# In[ ]:


df.Contp.value_counts()


# In[ ]:


df.Reg.value_counts()


# In[ ]:


df.Status.value_counts()


# In[ ]:


df.Stage.value_counts()


# In[ ]:


df.StageSub.value_counts()


# In[ ]:


df[df.Con == 'Bosnia and Herzegovina/Yugoslavia (former)']


# In[ ]:


df.Con.isin(['China','Palestine','Iran','Sri Lanka','USA','America'])


# In[ ]:


df.isna().sum().tolist()


# Since our dataset only has 1518 rows, we would incur too much of a dataloss to delete all the 734 nulls (almost 50%). So we'll be keeping it for now.

# In[ ]:


df.iloc[:,10:14]


# In[ ]:


df.index[12]


# What is the timeframe of our dataset?

# In[ ]:


print(df.Dat.min())
print(df.Dat.max())


# Doesn't look right, since we can see dates outside of this timeframe with our sampled data displayed above. Let's investigate further.

# In[ ]:


df.Dat.dtypes


# In[ ]:


df.Dat=pd.to_datetime(df.Dat)


# In[ ]:


df.Dat.dtypes


# In[ ]:


df.info()


# In[ ]:


df.dtypes.unique()


# In[ ]:


print(df.Dat.min())
print(df.Dat.max())


# In[ ]:


df['month']=df['Dat'].dt.year


# In[ ]:


df.month


# We've named the year column incorrectly as 'month', let's correct it:

# In[ ]:


df = df.rename(columns={'month' : 'year'})


# In[ ]:


# df.groupby(['year'])
# df.groupby(['Animal']).mean()
# df.year.sort_values().unique() # useless, we dont need the series YEAR sorted, we know 1990 is the min value and 2015 is max.
df.year.sort_values().value_counts() #sorting by value_counts
# df.year(sort=True).value_counts()


# In[ ]:


df.year.groupby(['year']).count()


# In[ ]:


df.groupby('year').year.value_counts() # only YEAR
# df.groupby('year').sum() # ALL columns


# Let's graph this.

# In[ ]:


df.groupby(df["year"]).count().plot(kind="bar") # this is using the newly created YEAR column directly
# df.groupby(df["Dat"].dt.year).count().plot(kind="bar") # this uses the datetime column 'Dat' and extracts YEAR from it.


# :) 
# 
# Badly proportioned plot, let's try to fix it:

# In[ ]:


(df.groupby('year').year.value_counts()).plot(kind='bar')
plt.show()


# In[ ]:


df.plot.bar(x=df.year, rot=0)

