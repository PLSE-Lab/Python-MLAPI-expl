#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
import math as mt
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Hello,
# I went to be a data scientist, and 2 studies I have done by doing several kynalktan researches are as follows. The data on the Kaggle side is taken from a developed kernel first. Thanks to my friend who works, eternally. ( https://www.kaggle.com/mczielinski/bitcoin-historical-data). The data explanations are as follows. 
# 
# CSV files for select bitcoin exchanges for the time period of Jan 2012 to July 2018, with minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC and indicated currency, and weighted bitcoin price. Timestamps are in Unix time. Timestamps without any trades or activity have their data fields forward filled from the last valid time period. If a timestamp is missing, or if there are jumps, this may be because the exchange (or its API) was down, the exchange (or its API) did not exist, or some other unforseen technical error in data reporting or gathering. All effort has been made to deduplicate entries and verify the contents are correct and complete to the best of my ability, but obviously trust at your own risk. 

# Then we load our data set
# 

# In[ ]:


df=pd.read_csv("../input/bitcoin.csv")


# We then check our data set (record, info, column based)
# 

# In[ ]:


df.info


# In[ ]:


df.corr()


# In[ ]:


df.shape


# After the data, we add a column where we calculate the opening and closing price differences in our data set, and we use the data to prepare the data set preparation

# In[ ]:


df["GAP"]=df.Open-df.Close


# In[ ]:


df2=df[df.GAP>0]


# In[ ]:


df2.shape


# After observing that our data set is too much data, we are reducing our data set considering that we will evaluate the data after 01/01/2018.

# In[ ]:


df3=(df2[df2.Timestamp>=1514764800])
df3.shape


# We first cycle through the data coming from the columns of our data set and look at the different data.

# In[ ]:


fig = plt.figure(figsize=(20,15))
cols = 5
rows = mt.ceil(float(df3.shape[1]) / cols)
for i, column in enumerate(df3.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df3.dtypes[column] == np.object:
        df3[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df3[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


# All Data 

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df3.Open.plot(kind = 'line', color = 'R',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df3.Close.plot(color = 'B',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('After_201801')            # title = title of plot
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df.Open.plot(kind = 'line', color = 'g',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Close.plot(color = 'r',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('All Data')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='GAP', y='Open',alpha = 0.5,color = 'red')
plt.xlabel('GAP')              # label = name of label
plt.ylabel('Open')
plt.title('Open-GAP')            # title = title of plot
plt.show()


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='GAP', y='Close',alpha = 0.5,color = 'B')
plt.xlabel('GAP')              # label = name of label
plt.ylabel('Close')
plt.title('Close-GAP')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='Timestamp', y='High',alpha = 0.5,color = 'Blue')
plt.xlabel('High')              # label = name of label
plt.ylabel('Timestamp')
plt.title('High-Timestamp')            # title = title of plot
plt.show()


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='Volume_(Currency)', y='Close',alpha = 0.5,color = 'red')
plt.xlabel('Volume_(Currency)')              # label = name of label
plt.ylabel('Close')
plt.title('Close-Volume_(Currency)')            # title = title of plot
plt.show()



# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df3.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


df3.Close.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:




