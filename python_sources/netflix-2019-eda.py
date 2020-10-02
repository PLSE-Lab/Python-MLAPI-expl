#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
cf.go_offline()


# In[ ]:


df = pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv',parse_dates=True)


# In[ ]:


df.head()


# In[ ]:


rating=df.groupby('rating',as_index=False)['show_id'].count()
rating['rs']=rating['rating'].map(rating_system)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='rating',y='show_id',data=rating)


# In[ ]:


df['rating'].unique()


# In[ ]:


def rating_system(x):
    if x in ('G', 'TV-Y', 'TV-G'):
        return 'Young Kids'
    elif x in ('PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG'):
        return 'Older Kids'
    elif x in ('PG-13', 'TV-14'):
        return 'Teens'
    elif x in ('R', 'NC-17', 'TV-MA'):
        return 'Adult'
    else:
        return 'Unrated'
    


# In[ ]:


df['rating_system']=df['rating'].map(rating_system)


# In[ ]:


df['rating_system'].unique()


# In[ ]:


plt.figure(figsize=(10,6))
sns.stripplot(x='rating',y='show_id',data=rating,hue='rs')


# In[ ]:


X = df['description']
y = df['rating_system']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


X = cv.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


nb.fit(X_train,y_train)


# In[ ]:


predictions = nb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:


def get_len(x):
    return len(x.split(','))


# In[ ]:


df['cast_len'] = df['cast'].dropna().apply(get_len)
df['reach']=df['country'].dropna().apply(get_len)


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(15,7))
sns.set_style('whitegrid')
plt.set_cmap('viridis')
sns.barplot(x=df['cast_len'],y=df['show_id'],data=df,estimator=np.size,hue=df['type'])
plt.ylabel('no of shows')


# In[ ]:


plt.figure(figsize=(15,7))
sns.set_style('whitegrid')
plt.set_cmap('viridis')
sns.barplot(x=df['reach'],y=df['show_id'],data=df,estimator=np.size,hue=df['type'])
plt.ylabel('no of shows')


# In[ ]:


sns.jointplot(x='cast_len',y='reach',data=df,color='purple')


# In[ ]:


df['length']=pd.to_numeric(df['dur'],errors='coerce')


# In[ ]:


df['country'].nunique()


# In[ ]:


df['dur']=df['duration'].apply(lambda X : X.split()[0]*1)


# In[ ]:


plt.figure(figsize=(10,8))
sns.lineplot(x=df[(df['type']=='Movie')&(df['release_year']>1999)].groupby('release_year',as_index=False).mean()['release_year'],y=df[(df['type']=='Movie')&(df['release_year']>1999)].groupby('release_year',as_index=False).mean()['length'])


# In[ ]:


df['date_added']


# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='release_year',y='show_id',data=df[df['release_year']>1990],estimator=np.size)


# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='release_year',y='length',data=df[df['release_year']>2007],estimator=np.mean)


# In[ ]:


plt.figure(figsize=(20,6))
df[df['type']=='Movie'][['title','length']].sort_values('length',ascending=False).head(20)


# In[ ]:


genre = df['listed_in']
genre.head(3)


# In[ ]:




