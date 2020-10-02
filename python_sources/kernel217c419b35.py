#!/usr/bin/env python
# coding: utf-8

# In[ ]:


../input/videogamesales/vgsales.csv# This Python 3 environment comes with many helpful analytics libraries installed
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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d


# In[ ]:


data=pd.read_csv("../input/videogamesales/vgsales.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


AvgNASls= data.groupby('Genre')['NA_Sales'].mean()
AvgNASls


# In[ ]:


SumNASls= data.groupby('Genre')['NA_Sales'].sum()
SumNASls


# In[ ]:


ax=sns.countplot(x = "Genre", data=data) 
plt.xticks(rotation=45)
ax.set_title(label='Number of Genres', fontsize=15)


# In[ ]:


ax=sns.barplot(x='Genre',y='NA_Sales', data=data) 
plt.xticks(rotation=90)
ax.set_title(label='Average NA_Sales', fontsize=15)


# In[ ]:


ax=sns.barplot(x='Genre',y='Global_Sales', data=data) 
plt.xticks(rotation=90)
ax.set_title(label='Average Global Sales', fontsize=15)


# In[ ]:


ax=sns.barplot(x="Year", y="Other_Sales", data=data)
plt.xticks(rotation=90)
ax.set_title(label='Average Other Sales', fontsize=15)


# In[ ]:


sns.barplot(x='Year',y='Global_Sales', data=data) 
plt.xticks(rotation=90)


# In[ ]:


sns.barplot(x="Year", y="Other_Sales", data=data,ci=20)
plt.xticks(rotation=90)


# In[ ]:


sns.barplot(x="Year", y="Other_Sales", data=data,ci=100)
plt.xticks(rotation=90)


# In[ ]:


sns.scatterplot("Rank","NA_Sales",data=data)


# In[ ]:


sns.scatterplot(x="Year", y="EU_Sales", data=data);


# In[ ]:


sns.scatterplot(x="Year", y="EU_Sales",hue='Genre',data=data);


# In[ ]:


sns.catplot(x="Genre", y="NA_Sales",data=data)
plt.xticks(rotation=90)


# In[ ]:


sns.catplot(x="Genre", y="NA_Sales",hue='Year',data=data)
plt.xticks(rotation=90)


# In[ ]:


sns.catplot(x="Genre", y="Year", kind="box", data=data)
plt.xticks(rotation=45);


# In[ ]:


sns.catplot(x='Year',y='NA_Sales', col='Genre',col_wrap=3,data=data,kind='bar',height=4,aspect=2)


# In[ ]:


ax=sns.lineplot(x="Year", y="Global_Sales",ci=None, data=data);
ax.set_title(label='Global sales/year', fontsize=15)


# In[ ]:


sns.lineplot(x="Year", y="Global_Sales",hue='Genre', data=data);


# In[ ]:


sns.lineplot(x="Genre", y="NA_Sales",hue='Year', data=data)
plt.xticks(rotation=45);


# In[ ]:




