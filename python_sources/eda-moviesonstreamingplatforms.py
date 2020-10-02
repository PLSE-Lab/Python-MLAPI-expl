#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


im= pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')


# In[ ]:


im.head()


# In[ ]:


im.shape


# In[ ]:


im.describe()
#data.describe() function will only display numarical data
#here we can see counts, max value etc and we can see the missing values also in count section 


# In[ ]:


im.info()
#data.info provides counts of all columns and also display type of data


# In[ ]:


sns.heatmap(im.isnull())
#visually we see the missing values using heatmap 


# # Data Visualization

# In[ ]:


sns.countplot(x='Netflix', data=im)


# In[ ]:


im["Netflix"].value_counts()
# counts of values


# In[ ]:


sns.countplot(x='Hulu', data=im)


# In[ ]:


im["Hulu"].value_counts()


# In[ ]:


sns.countplot(x='Prime Video', data=im)


# In[ ]:


im["Prime Video"].value_counts()


# In[ ]:


sns.countplot(x='Disney+', data=im)


# In[ ]:


im["Disney+"].value_counts()


# In[ ]:


sns.countplot(x='Age', data=im)


# In[ ]:


sns.pairplot(im)
fig=plt.gcf()
fig.set_size_inches(20,20)
#as most of them are just categorical data so correlation is not there 


# In[ ]:


sns.scatterplot(x="Year", y="Runtime", hue="Age",data=im)
fig=plt.gcf()
fig.set_size_inches(10,10)
#we can see the outlier and the hue is Age. we can see that more films are made as time proceeds, this can be VISUALIZED using distribution graph


# In[ ]:


sns.distplot(im['Year'])
#by seeing the distribution graph we can say that most movies are made in period of 2000 to 2020 than 1940 to 2000


# In[ ]:


plt.figure(figsize=(15,7))
chains=im['Language'].value_counts()[:20]#change this value to see more result
sns.barplot(x=chains,y=chains.index,palette='Set2')
plt.title("Languages most commonly made ",size=20,pad=20)
plt.xlabel("Counts",size=15)
# English movies are made more in world and it followed by Hindi movies 


# In[ ]:


plt.figure(figsize=(15,7))
chains=im['Directors'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most movies made by Director ",size=20,pad=20)
plt.xlabel("Counts",size=15)


# In[ ]:


plt.figure(figsize=(15,7))
chains=im['Genres'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set2')
plt.title("Genres",size=20,pad=20)
plt.xlabel("Counts",size=15)
#it looks like people like more Drama movies compared to action movies


# In[ ]:


plt.figure(figsize=(15,7))
chains=im['Country'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set3')
plt.title("Most movies made by Country",size=20,pad=20)
plt.xlabel("Counts",size=15)
#here we can see that USA makes more movies and follwed by India


# In[ ]:


sns.distplot(im['IMDb'], bins=20)
#this is the distribution graph of IMDb rating but we cant campare to Rotten Tomatoes because if see the heatmap
#there are so missing values, so we drop NaN values and compare.

#so about this distribution graph we can see avrage is about 6 rating


# In[ ]:


# Here Rotten Tomatoes is object value so we have to convert it to float. we do that by removing % symbol 


# In[ ]:


im_copy = im.copy(deep = True)


# In[ ]:


im_copy['Rotten Tomatoes'].unique()
im_copy= im_copy.loc[im_copy['Rotten Tomatoes'] !='NEW']
im_copy=im_copy.loc[im_copy['Rotten Tomatoes'] !='-'].reset_index(drop=True)
remove_slash = lambda x:x.replace('%','') if type(x)==np.str else x
im_copy['Rotten Tomatoes']=im_copy['Rotten Tomatoes'].apply(remove_slash).str.strip().astype('float')
im_copy['Rotten Tomatoes'].head()


# In[ ]:


im_copy.isnull().sum()
im_copy.dropna(how='any', inplace=True)


# In[ ]:


im_copy.shape
# initally 16744 rows, after dropna there are only 3301


# In[ ]:


sns.distplot(im_copy['IMDb'])


# In[ ]:


sns.distplot(im_copy['Rotten Tomatoes'])


# In[ ]:


# clearly we can see that distribution graph are not same that means rating differs for same movies

