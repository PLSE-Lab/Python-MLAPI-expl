#!/usr/bin/env python
# coding: utf-8

# 
# # **BITCOIN HISTORICAL DATASET**
# ![](https://cdn.iconscout.com/icon/free/png-256/bitcoin-385-920570.png)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization tool
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#I read the data and i brought the titles.
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
df.head()


# In[ ]:


df.info()
#I'm looking at what information I can get about the dataframe.
#I see that Timestamp, Open, High, Low, Close, Volume_(BTC), Volume(Currency) and Weighted_Price ---> headers of dataset


# In[ ]:


df.corr()

#I look at the correlation values between the headings.


# In[ ]:


f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df.corr(), vmin=0, vmax=1, cmap="YlGnBu", center=None, robust=False, annot=True, fmt='.1g', annot_kws=None, linewidths=0.2, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None)
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
df.Open.plot(kind = 'hist',bins = 100,figsize = (10,10),color="b")
plt.show()


# In[ ]:


# Line Plot 
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df.Open.plot(kind = 'line', color = 'r',label = 'Open',linewidth=3,alpha = 0.6,grid = True,linestyle = ':')
df.Close.plot(color = 'b',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = Open, y = Close
df.plot(kind='scatter', x='Open', y='Close',alpha = 0.5,color = 'blue')
plt.xlabel('Open')              # label = name of label
plt.ylabel('Close')
plt.title('Open - Close with Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


df.plot(kind='scatter', x='Volume_(BTC)', y='Weighted_Price',alpha = 0.5,color = 'blue')
plt.xlabel('Volume_(BTC)')              # label = name of label
plt.ylabel('Weighted_Price')
plt.title('Volume vs Weighted Price')            # title = title of plot
plt.show()


# In[ ]:


#Scatter Plot | Volume_(BTC) VS Volume_(Currency)
df.plot(kind = 'scatter', x = 'Volume_(BTC)', y = 'Volume_(Currency)', alpha = 0.5, color='g')

plt.xlabel('Volume_(BTC)')
plt.ylabel('Volume_(Currency)')
plt.title('Scatter Plot | Volume_(BTC) VS Volume_(Currency)')
plt.show()

