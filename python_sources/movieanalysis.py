#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd   #for data preprocessing
import seaborn as sns 
import matplotlib.pyplot as plt #for data visualisation
from collections import Counter


# **Reading In Data**

# In[ ]:


data= pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')


# In[ ]:


data.head()    #To display 5 data sets to get an idea of how the data looks like 


# In[ ]:


data.dtypes


# In[ ]:


data.describe()


# Finding The Most Popular Country Among The Country(Country involved in a lot of Movies)

# In[ ]:


plt.figure(1, figsize=(20,20))
plt.title('Most Occuring Coutry')
xlabel='Countries'
ylabel='Frequency'
sns.countplot(x='country', order=data['country'].value_counts().index[0:20], data=data)


# Most Popular TV-Rating

# In[ ]:


plt.figure(1,figsize=(20,20))
plt.title('TV Ratings')
xlabel= 'rating'
ylabel= 'frequency'
sns.countplot(x="rating", order=data['rating'].value_counts().index[0:20], data=data)


# From the analysis done above, it can be observed that TV-MA( which has a lot of mature content is the most produced.
# Followed by TV-14 and TV-PG. 
# The plot above shows the necessary infromation

# In[ ]:


plt.figure(1, figsize=(15,10))
plt.title('TV Shows vs Movies')
data['type'].value_counts().plot(kind='pie')


# In[ ]:


data['rating'].value_counts().plot(kind='pie', figsize=(15,10), pctdistance=0.8, autopct='%1.2f%%' )


# In[ ]:


plt.figure(1, figsize=(10,10))
plt.title('Common Genre')
xlabel=('Genre')
ylabel=('Frequency')
sns.countplot(x='listed_in', order= data['listed_in'].value_counts().index[0:15], data=data)
plt.xticks(rotation=60)


# Most Popular Genre or Most Produced Genre

# In[ ]:


c=Counter(data['listed_in'])
genre=c.most_common(15)
movie=[]
count=[]
for i in genre:
    movie.append(i[0])
    count.append(i[1])
    
movie.reverse()
count.reverse()
    
print(movie)
print(count)

# print(c)


# In[ ]:


plt.title('Most Popular Genre')
xlabel=('genre')
ylabel=('Number of movies produced')
plt.figure(1, figsize=(15,15))
plt.barh(movie, count)


# In[ ]:


plt.figure(1, figsize=(15,15))
plt.title('Most Popular Actor')
xlabel= 'Actors'
ylabel= 'Number of occurence'

sns.countplot (x='cast', order=data['cast'].value_counts().index[0:15], data=data)
plt.xticks(rotation=30)

