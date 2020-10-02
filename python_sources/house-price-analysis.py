#!/usr/bin/env python
# coding: utf-8

# <h1><u>**Introduction**</u></h1>
# 
# <h4>I tried to illustrate a relationship between Area and Price from the House Pricing database.
# I first of all analysed the data and then make a conclusion of what kind of dependency this two variables have between them.
# 
# Hope you'll like it.</h4>

# <h2><u>**Analysis** </u></h2>

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


# > <h1>**IMPORTING DATASET**</h1>

# In[ ]:


##IMPORTING OF CSV DATASET
url='../input/HousePrices_HalfMil.csv'
df=pd.read_csv(url)
df.head()


# <h1> **Understanding of Data** </h1>
# 
# <h4> After Importing the dataset we now try to understand the data.</h4>

# In[ ]:


df.describe(include='all')


# In[ ]:


#Cheacking Of Null Datas
df.info()


# <h1>Data Wrangling</h1>

# *   <h2>****Boxplot between Area and Price****</h2>

# In[ ]:


###Boxplot to show us the trend or the relation between Area and Price
sns.boxplot(x='Area',y='Prices',data=df)


# * <h2> Scatter Plot </h2>

# In[ ]:


x=df['Area']
y=df['Prices']
plt.scatter(x,y)
plt.title('Area Vs Price /n Scatter Plot')
plt.xlabel('Area')
plt.ylabel('Prices')


# <h1> **Conclusion** </h1>
# 
# <h4> Boxplot and Scatterplot shows that increase in Area is proportional to the Price increase(Linear Relation : Positive). 
# That means if area is increased then price will get increased or vice-versa. </h4>

# In[ ]:




