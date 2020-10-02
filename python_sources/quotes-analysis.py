#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#pd.options.display.max_colwidth = 100
pd.set_option('display.max_colwidth', -1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


quotes = pd.read_json('../input/quotes.json')
quotes.head()


# In[ ]:


#Check if there is NaN Values in Columns
quotes.isnull().sum()


# In[ ]:


#Check if there is NaN Values in Rows
quotes.isnull().sum(axis=1)


# **Data is Cleaned as we see in the row and columns of the data**

# In[ ]:


#Convert the Popularity Column into Some More Precise Values or Change it from 0.1 to 1.0
quotes['Popularity'] = quotes['Popularity']*10


# In[ ]:


#Check How Many Categories We Have
categories = quotes['Category']
cat = categories.unique()
cat


# In[ ]:


#Now We Calculate the Popularity of Author that is most in all
group = quotes.groupby('Author')['Popularity'].mean()


# In[ ]:


ans = group.sort_values(ascending=False).head()#Top Author is William W. Purkey
ans


# In[ ]:


#Check the Quotes According to Any Category
pd.DataFrame(quotes.loc[quotes['Category']=='inspiration']['Quote']).head()


# In[ ]:


#Check Mean of All Categories
popular = quotes.groupby('Category')['Popularity'].mean()
popular


# In[ ]:


#Plot Graph of all Categories and Their Mean Values
fig = plt.gcf()
fig.set_size_inches(40, 10.5, forward=True)
plt.bar(popular.index,popular.values)


# In[ ]:


#Top 5 Categories
popular.nlargest(5)


# In[ ]:


#Split the First 3 Tags in different Columns
quotes['Tags'] = quotes['Tags'].apply(lambda x: str(x).replace('-',' '))
quotes['Tags'].unique()


# In[ ]:


quotes['Tag_1'] = ''
quotes['Tag_2'] = ''
quotes['Tag_3'] = ''

quotes['Tag_1']= quotes['Tags'].apply(lambda x: str(x).split(',',3)[:1])
quotes['Tag_2']= quotes['Tags'].apply(lambda x: str(x).split(',',3)[1:2])
quotes['Tag_3']= quotes['Tags'].apply(lambda x: str(x).split(',',3)[2:3])

quotes['Tag_1'] = quotes['Tag_1'].str.get(0)
quotes['Tag_2'] = quotes['Tag_2'].str.get(0)
quotes['Tag_3'] = quotes['Tag_3'].str.get(0)

quotes['Tag_1'] = quotes['Tag_1'].apply(lambda x: str(x).replace("'",""))
quotes['Tag_1'] = quotes['Tag_1'].apply(lambda x: str(x).replace("[",""))

quotes['Tag_2'] = quotes['Tag_2'].apply(lambda x: str(x).replace("'",""))
quotes['Tag_3'] = quotes['Tag_3'].apply(lambda x: str(x).replace("'",""))


quotes.head()


# In[ ]:


#Check Tag_1 Popularity
Tag1_popularity = quotes.groupby('Tag_1')['Popularity'].mean()


# In[ ]:


#Top 5 in Tag_1
Tag1_popularity.nlargest(5)


# In[ ]:




