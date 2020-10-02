#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/chess/games.csv')
data.info()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5,fmt='.1f',ax=ax) # annot karelerin uzerindeki degerleri gosterir
plt.show()                                                          # ax sol ve asagidaki degerler


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


beyaz ve siyah oyuncularin kuvvet grafigi;


# In[ ]:


data.white_rating.plot(kind = 'line',color = 'g', label = 'white_rating', linewidth =1, alpha = 0.5, grid = True, linestyle = ':')
data.black_rating.plot(color = 'r', label ='black_rating', linewidth = 1, alpha = 0.5, grid = True,linestyle = '-.')
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# beyaz ve siyah oyuncularin kuvvet dagilimi;

# In[ ]:


data.plot(kind = 'scatter', x = 'white_rating', y= 'black_rating', alpha = 0.5, color = 'red')
plt.xlabel('white_rating')
plt.ylabel('black_rating')
plt.title('White Black Rating Scatter Plot')
plt.show()


# kac hamlede oyun bitmis bunu gosterelim;

# In[ ]:


data.turns.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# kuvveti 2000 uzeri oyun oyuncularin oynadigi maclari gosterelim;

# In[ ]:


x = data['white_rating']>2000
y = data['black_rating']>2000
a = x&y
data[a]


# ayni sekilde yukaridakinin np ile kullanimi

# In[ ]:


data[np.logical_and(data['white_rating']>2200, data['black_rating']>2200 )]


# In[ ]:


for index,value in data[['black_rating']][0:1].iterrows():
    print(index," : ",value)

