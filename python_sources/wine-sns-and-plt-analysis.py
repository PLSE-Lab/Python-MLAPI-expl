#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/winemag-data_first150k.csv")


# In[ ]:


data


# In[ ]:


data.corr()


# In[ ]:


data.describe()


# In[ ]:


data.points = data.points.astype(float)


# In[ ]:


data.info()


# In[ ]:


sns.countplot(data['points'])


# In[ ]:


sns.distplot(data['points'])


# In[ ]:


wine_level = []
wine_level = data.points/data.price
data['wine_level'] = data.points/data.price
sns.countplot(data['wine_level'])


# In[ ]:


sns.kdeplot(data.query('price < 100').price)


# In[ ]:


sns.kdeplot(data.query('wine_level > 4.1').price) #wine_level mean = 4.107469


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x=data['price'], y=data['points'])
plt.xticks(rotation= 45)
plt.xlabel('price')
plt.ylabel('points')
plt.title('Price And Points Plot')


# In[ ]:


data["wine_level"].value_counts().head(10).plot.bar()


# In[ ]:


data['country'].value_counts().head(10).plot.bar()


# In[ ]:


data['wine_level'].value_counts().sort_index().plot.bar()

