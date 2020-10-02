#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import datetime
import wordcloud
import json
from datetime import datetime
import math


# # Loading and Preparing

# ## Import US vidoes Dataset

# In[ ]:


df = pd.read_csv("../input/youtube-new/USvideos.csv")
import json
cat_list=json.loads( open('../input/youtube-new/US_category_id.json').read())['items']
cat_dict = {}
for w in cat_list:
    cat_dict[int(w['id'])] = w['snippet']['title']


# ## Adding useful features

# In[ ]:


df['category']= df['category_id'].apply(lambda a:cat_dict[a])
df['n_tags'] =df['tags'].apply(lambda w: w.count('|')+1)
df['trending_date_object'] = df['trending_date'].apply(lambda d:datetime.strptime(d,"%y.%d.%m"))
df['publish_time_object'] = df['publish_time'].apply(lambda d:datetime.strptime(d,"%Y-%m-%dT%H:%M:%S.%fZ"))
df['trending_delta_object']= df['trending_date_object']-df['publish_time_object']
df['trending_delta'] = df['trending_delta_object'].apply(lambda w:math.ceil(w.days*24+w.seconds/3600))


# *View a gist of the dataset*

# In[ ]:


df.head().transpose()


# In[ ]:


num_attribs = ['views','likes','dislikes','comment_count','n_tags','trending_delta']
cat_attribs= ['tags','description','title','publish_time','trending_date',
             'comments_disabled','ratings_disabled','video_error_or_removed','category',
             'channel_title','trending_delta_object',
              'trending_date_object','publish_time_object']


# ## Preperation for data cleaning
# 

# adding explicit null values to use for imputing values with mean

# In[ ]:


rem_count = 100
rem_attribs = ['views','likes','dislikes','comment_count']
for r in rem_attribs:
    d = df[r].copy()
    d.loc[list(np.random.randint(0,len(df),rem_count))] =np.NaN
    df[r] = d


# # Data Cleaning

# In[ ]:


df.info()


# **Cat Attribs which have missing values**
# * Description - Has *36912* missing values
# 
# **Num Attribs which have missing values**
# * Views has *100* missing values (was et explicity before)
# * Likes has *100* missing values (was et explicity before)
# * Dislikes has *100* missing values (was et explicity before)
# * Comment_count has *100* missing values (was et explicity before)
# 

# ## Categorical Data Cleaning

# fill null values in description column with ""

# In[ ]:


#Description has null values 
df["description"] = df["description"].fillna(value="")


# ## Numerical Data Cleaning

# In[ ]:


df[rem_attribs] =df[rem_attribs].fillna(df[rem_attribs].mean())


# In[ ]:


df.info()


# # Visualizing Data

# ## Numerical 

# ### Scatter Plot

# In[ ]:


from pandas.plotting import scatter_matrix
done = []
for i in num_attribs:
    for j in num_attribs:
        if (i!=j and (j,i) not in done):
            print(i,j)
            done.append((i,j))
            plt.figure(figsize=(8,5))
            plt.scatter(df[i],df[j])
            plt.title(i+" "+j)
            plt.xlabel(i)
            plt.ylabel(j)


# ### Histograms

# In[ ]:


num_attribs


# In[ ]:


for i,w in enumerate(num_attribs):
    _max = df[w].max()
    _min = df[w].min()
    _a,_b = df[w].describe()[['25%','75%']]
    _width=  2*(_b-_a)/(len(df)**(1/3))
    bins =math.floor( (_max-_min)/_width)
    print(bins,w)
    plt.figure(figsize=(8,5))
    plt.title(w)
    df[w].hist(bins=bins)
    if i==0:
        plt.xlim(0,0.25e8)
        plt.ylim(0,2500)
    elif i==1:
        plt.xlim(0,500000)
        plt.ylim(0,2000)
    elif i==2:
        plt.xlim(0,10000)
        plt.ylim(0,4000)
    elif i==3:
        plt.xlim(0,50000)
        plt.ylim(0,3500)
    elif i==5:
        plt.xlim(0,3000)
        plt.ylim(0,2000)
    


# ### Box Plots

# In[ ]:


num_attribs


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,TransformerMixin
scaled_data = MinMaxScaler().fit_transform( df[num_attribs])


# Trimming data since there are a lot of outliers

# In[ ]:


class Trim(BaseEstimator,TransformerMixin):
    def __init__(self,t):
        self.t = t
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X =np.sort(X,axis=0)
        return np.apply_along_axis(trim,axis=0,arr=X,t=self.t)
    
def trim(data,t=0.15):
    return data[ int(len(data)*t):int(len(data)*(1-t)) ]


# In[ ]:


from sklearn.pipeline import Pipeline
p = Pipeline([
    ('trim',Trim(0.15)),
    ('scale',MinMaxScaler())
])
scaled_data = p.fit_transform(df[num_attribs])

plt.figure(figsize=(15,10))
_=plt.boxplot(scaled_data)
_=plt.xticks([1,2,3,4,5,6],labels=num_attribs)


# In[ ]:




