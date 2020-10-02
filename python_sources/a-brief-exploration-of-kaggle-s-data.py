#!/usr/bin/env python
# coding: utf-8

# ### Kaggle Data Sets Review
# 
# _This is a brief analysis of a set of datasets from  [Kaggle](https://www.kaggle.com/)._
# 
# _This is the first analysis I did with little tutorial support. It is very simple, but for me it is already a great advance._
# 
# 
# 

# In[41]:


#importing Pandas. 


import pandas as pd


# In[71]:


# loading the dataset

kaggle = pd.read_csv(r'../input/kaggle_datasets.csv')


# ## knowing the data!
# 
# **Here we will know the size of the data set your columns and content. It is a very useful and important phase in the analysis process.
# **
# 
# _Lets do this!_

# In[43]:


#Displaying the first few lines of Dataset
kaggle.head()


# In[44]:


# Discovering the type of the dataset. In this case, the pandas recognize it as a DataFrame. It's basically a table!

type(kaggle)


# In[45]:


#The Shape property shows the size of the dataset. This dataset has 8036 rows and 14 columns

kaggle.shape


# In[46]:


# Showing the columns
kaggle.columns


# ## Using the describe() function 
# 
# 
# _This function shows a statistical summary of columns with numeric data. Includes the 'include = all' argument to see even useful information about columns with categorical data._
# 
# _We have already been able to extract some interesting information, for example, the user with the largest number of published datasets is the [NLTK Data](https://www.kaggle.com/nltkdata), with 94 datasets!_

# In[49]:



kaggle.describe(include='all')


# In[50]:


# Here we use the count () method, allows us to see if all the columns are completely filled

kaggle.count()


# In[51]:



pd.value_counts(kaggle['views'])


# In[52]:


#There are only 20 datasets with 0 views on the Kaggle website

pd.value_counts(kaggle['views'] == 0)


# In[53]:


#Using .loc (), we can know what datasets are 0 views
kaggle.loc[kaggle['views']==0]


# In[54]:


#Here we are seeing datasets with more than 100 views and less than 100 downloads
kaggle.loc[(kaggle['views'] >= 100) & (kaggle['downloads'] <= 100)]


# In[55]:


#Dataset with highest number of views
kaggle.loc[kaggle['views'] == 429745]


# In[56]:


# the 10 datasets with the highest number of views


kaggle.sort_values(['views'], ascending = False).head(10)


# In[57]:


# the 10 datasets with the highest number of downloads (which are the ones with the largest number of views)

kaggle.sort_values(['downloads'], ascending = False).head(10)


# In[58]:


kaggle[kaggle['kernels']== 1].count()


# In[59]:


kaggle.nunique()


# In[62]:


#This chart shows the distribution of datasets considering views and downloads. It is clear that most datasets
# has less than 10k downloads
# Only 4 datasets have more than 300k views
kaggle.plot(figsize = (10,10), kind = 'scatter', x = 'views', y = 'downloads')


# In[61]:


#These are from Datasets with more than 300k views
kaggle.loc[kaggle['views'] > 300000]


# In[63]:


#Here we plot the obvious relationship between number of views and downloads. Using a heatmap. This explains why the same datasets appear in top views and downloads.

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))
sns.heatmap(kaggle.corr(), annot = True, fmt = '.2f', cbar = True)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)


# In[65]:


# What are the biggest contributors?
top_owner = pd.value_counts(kaggle['owner']).head(10)


# In[66]:


top_owner


# In[67]:


type(top_owner)


# In[68]:


top_owner.plot(figsize = (12,8), kind = 'bar', title = 'Top Contributors')


# In[34]:


# Out of curiosity, a little analysis of the datasets of the user Jacob Boysen. He posted some very interesting sets.

kaggle.loc[kaggle['owner'] == 'Jacob Boysen']


# In[35]:


#To facilitate my analysis, I created a new variable with Jacob's data


jacob = kaggle.loc[kaggle['owner'] == 'Jacob Boysen']


# In[37]:


#Seeing the 10 datasets with more views than Jacob posted

jacob.sort_values(['views'], ascending = False).head(10)


# In[38]:


jacob.count()


# In[39]:


# Here we see the 29 Jacob datasets sorted by number of views

top10 = jacob[['title', 'views']].set_index('title').sort_values('views', ascending=True)
top10.plot(kind = 'barh',figsize = (15,15), grid = True, color = 'blue', legend = True)


# **This was a brief review. Actually a data explorer, I was able to use some nice functions and it was quite fun to realize that I'm beginning to understand better how to do analysis using Python. This was only a small step towards the really complex analysis of Data Science.**
# 
# _Please help me with tips, suggestions or corrections!_
# 
# _Thanks!_

# In[ ]:




