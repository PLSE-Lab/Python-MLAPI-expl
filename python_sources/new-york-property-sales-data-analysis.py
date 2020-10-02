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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df= pd.read_csv('../input/nyc-rolling-sales.csv')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.drop(['EASE-MENT','APARTMENT NUMBER','Unnamed: 0','ZIP CODE'],axis=1, inplace=True)


# First, let's convert few columns which are categorical into integer values, so that it is easy for machine to handle values.

# In[ ]:


def sale(string):
    if ('-' in string):
        return np.NaN
    else:
        return int(string)
def year_built(string):
    if string==0:
        return np.NaN
    elif ('-' in string):
        return np.NaN
    else:
        return int(string)


# In[ ]:


df['SALE PRICE']=df['SALE PRICE'].map(lambda x: sale(x))


# In[ ]:


df['YEAR BUILT']=df['YEAR BUILT'].map(lambda x: int(x))


# In[ ]:


df['YEAR SOLD']=df['SALE DATE'].map(lambda x: x[0:4])


# In[ ]:


df['YEAR SOLD']=df['YEAR SOLD'].map(lambda x: int(x))


# In[ ]:


df['MONTH SOLD']=df['SALE DATE'].map(lambda x: x[5:7])


# In[ ]:


df['MONTH SOLD']=df['MONTH SOLD'].map(lambda x: int(x))


# In[ ]:


df['GROSS SQUARE FEET']=df['GROSS SQUARE FEET'].map(lambda x: year_built(x))
df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].map(lambda x: year_built(x))


# In[ ]:


df['YEAR BUILT'][df['YEAR BUILT']==0]=np.NaN


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


df.dropna(axis=0,inplace=True)


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# **BOROUGH: A digit code for the borough the property is located in; in order these are Manhattan (1), Bronx (2), Brooklyn (3), Queens (4), and Staten Island (5).**

# In[ ]:


sns.countplot('BOROUGH',data=df,palette='Set2')
plt.title('Sales per Borough')


# * Maximum properties are sold in Queens followed by Staten Island

# In[ ]:


sns.barplot(y='RESIDENTIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)
plt.title('Sales per borough_Residential')


# * Residential Units are mainly sold in Manhattan

# In[ ]:


sns.barplot(y='COMMERCIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)
plt.title('Sales per borough_Commercial')


# * Manhattan has the most Commercial Units sold

# In[ ]:


sns.countplot(x='YEAR SOLD', data=df, palette='rainbow')
plt.title('Sales Rate from 2016-2017')


# In[ ]:


sns.barplot(x='YEAR SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)
plt.title('Sales per Borough from 2016-2017')


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x='MONTH SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)
plt.title('Sales per Borough from 2016-2017')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))


# In[ ]:


plt.figure(figsize=(20,5))
sns.countplot('MONTH SOLD', hue='YEAR SOLD', data=df, palette='RdBu_r')


# 
# * It is noticed that though the number of sales increased from  the year 2016 to 2017, the sales prices per Borough(location) remained in the same ranges
# * Also, the property prices are much higher at Manhattan than at any other location.
# * As per months, property sales for 2017 took place from January till August, and for 2016 from September till December.

# In[ ]:




