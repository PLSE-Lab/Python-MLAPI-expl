#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# ### Let's check 1st file: ../input/zomato.csv

# In[ ]:


nRowsRead = None # specify 'None' if want to read whole file
# zomato.csv has 51717 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/zomato.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'zomato.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


data =df1.copy()


# In[ ]:


data.loc[:,['address','location','listed_in(city)']].sample(8,random_state=7)


# In[ ]:


target = ['address','location','listed_in(city)']
for i in target:
    print(f'Number of unique values for {i} :  {data[i].nunique()}')


# In[ ]:


print(data['listed_in(city)'].unique())


# In[ ]:


data.head(5)


# In[ ]:


data.loc[:,'rate'] #index based search 


# In[ ]:


data[0:4:] #data.head(4)


# In[ ]:


data.index


# In[ ]:


data.loc[1] #loc is used to fetch value from index here 1 is first row  data.loc[0:3]


# In[ ]:


data.columns


# In[ ]:


data.loc[0:3,['url','name']] #get 4 rows and url,name column we can also do data.loc[0:3,'url':'']


# In[ ]:


data.iloc[1] == data.loc[1] #both are simmilar but use loc for label based indexing and iloc for index based indexing


# In[ ]:


data[data.votes == np.nan]


# In[ ]:


data.iloc[28,:]


# In[ ]:


data[pd.isna(data.dish_liked) == True] #show all where dish_liked is NaN


# In[ ]:


data.rate = data.rate.str.split('/',expand = True)[0]
data.rate = data.rate.fillna(1)
data[data.rate == 'NEW'] = 0
data[data.rate == '-'] = 0
data.rate = data.rate.astype('float64')


# In[ ]:


total_votes = data.rate.multiply(data.votes)
data['total_rating'] = total_votes
location_split = data.groupby('location')
location_split.first() #Print First Entry Of All Of The Groups Formed


# In[ ]:


rating_location = location_split.total_rating.sum() #use reset_index() to not make location as index
rating_location


# In[ ]:


rating_location.plot.bar(figsize = (20,10))


# In[ ]:




