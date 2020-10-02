#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/NBA_player_of_the_week.csv")


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.Age.plot(kind='line', color='g', label='Age', linewidth=2,alpha=0.5, grid=True,linestyle=':')
data["Seasons in league"].plot(color='r',label='Draft Year',linewidth=5, alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('Seasons in league')
plt.ylabel('Age')
plt.title('Line Plot')
plt.show()


# In[ ]:


data.plot(kind='scatter', x='Age', y='Season short',alpha=0.4, color='purple')
plt.xlabel('Age')
plt.ylabel('Season short')
plt.title('Age Draft Year Scatter Plot')


# In[ ]:


data.Age.plot(kind='hist', bins=30, figsize=(10,10))
plt.show()


# In[ ]:


data.Age.plot(kind='hist',bins=50)


# In[ ]:


series = data["Season short"]
print(series)


# In[ ]:


data_frame = data[['Season short']]
print(type(data_frame))


# In[ ]:


x = data['Season short']>2017
data[x]


# In[ ]:


data[np.logical_and(data['Season short']>2017,data['Draft Year']>2011)]


# In[ ]:


data[(data['Age']<21)&(data['Draft Year']>2012)]


# In[ ]:


for index,value in data[['Player']][0:20].iterrows():
    print(index,":",value)


# In[ ]:





# In[ ]:





# In[ ]:




