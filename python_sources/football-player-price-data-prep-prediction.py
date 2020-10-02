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


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


df=pd.read_csv('../input/top250-00-19.csv')


# In[ ]:


df.head()


# In[ ]:


# Checking the position of the players 
df['Position'].value_counts()


#  Lets see the position Defender, Sweeper and forward if we can place them in the more populated positions
#  Using google we tried to find most relative position to remove outliers 
# 

# In[ ]:


df[df['Position']=='Defender']
#Sergio Hellings Central midfielder


# In[ ]:


df[df['Position']=='Midfielder']


# In[ ]:


df[df['Position']=='Forward']


# In[ ]:


#Didier Martel midfilder
#Patricio Camps Attacking midfilder
#Mazhar Abdelrahman  central forward
#Sergio Hellings Central midfielder
#Tony Dinning = Defensive Midfield

df['Position']=np.where(df['Position']=='Forward', 'Central Midfield', df['Position'])
df['Position']=np.where(df['Position']=='Midfielder', 'Defensive Midfield', df['Position'])
df['Position']=np.where(df['Position']=='Defender', 'Central Midfield', df['Position'])
df['Position']=np.where(df['Position']=='Sweeper', 'Defensive Midfield', df['Position'])


# In[ ]:


df['Position'].value_counts()


# In[ ]:


# Visual Representation

plt.figure(figsize=(10,10))
sns.barplot(df['Position'].index, df['Position'].values)
plt.title('Count of Position in the Dataset')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('city', fontsize=12)
plt.show()


# In[ ]:




