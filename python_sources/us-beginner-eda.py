#!/usr/bin/env python
# coding: utf-8

# # US-Youtube Beginner EDA
# 
# Hi,
# 
# after just learning the basics of datascience I want to share my first notebook with you guys.
# I was heavily inspired by the awesome work in this notebook: https://www.kaggle.com/kabure/extensive-usa-youtube-eda
# and tried to include some of my own ideas and coding style.
# 
# Please feel free to comment and make suggestions where my conclusions or code might not be optimal :)
# 
# The questions we're trying to answer are very basic:
# * which are the most prominent categories on youtube?
# * which videos get the most views?
# * which videos have the highest engagement rate?
# * is there a best time to share new content (hour, day, month)?

# In[ ]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.json import json_normalize

pd.pandas.set_option('display.max_columns',None)


# In[ ]:


## import and normalize json data to get categories for each id
with open('../input/youtube-new/US_category_id.json') as f:
    d = json.load(f)

df_cat = pd.json_normalize(d['items'])[['id','snippet.title']]
df = pd.read_csv('../input/youtube-new/USvideos.csv')

## change dtype and prepare to join data on id
df['category_id'] = df['category_id'].astype(int)
df_cat['id'] = df_cat['id'].astype(int)
df = df.join(df_cat.set_index('id'), on='category_id')

## rename and drop unnecessary column
df.rename(columns={'snippet.title':'category'}, inplace=True)
df.drop('category_id', axis=1, inplace=True)


# # General Overview

# In[ ]:


print(f'DF Shape: {df.shape} \n')
print(df.nunique())


# In[ ]:


df.head(3)


# In[ ]:


df.describe()


# **Missing Values**

# In[ ]:


features_nan = [f for f in df.columns if df[f].isnull().sum() > 1]
for feature in features_nan:
    print('Feature:',feature, np.round(df[feature].isnull().mean()*100,2),'% of values missing')


# **Types of Variables**

# In[ ]:


num_feat = [f for f in df.columns if df[f].dtype in ['int64','float64']]
cat_feat = [f for f in df.columns if df[f].dtype in ['object','bool']]

print(f'Number of numerical variables: {len(num_feat)}')
print(f'Number of categorical variables: {len(cat_feat)}')


# After successfully loading and combining the datasets, we can take a brief look on the data at hand:
# * we're working with 40949 rows and 16 columns
# * by looking at the summary (df.describe()) it becomes obvious that we have to deal with a heavily (right) **skewed** dataset (mean > median / high max values)
# * dataset is **nearly** **complete**, only one variable has 1.39% of data missing
# * the dataset contains **4 numerical** and **12 categorical variables**
# 
# Since we want to get a deeper understanding which videos get the most views and engagement, we will taker a closer look at the numerical variables and the distribution

# # Numerical Variables

# In[ ]:


## check for skewness since outliers are present as seen in general overview
plt.figure(figsize=(14,6))

for idx,feature in enumerate(num_feat):
    print('Skewness of',feature, df[feature].skew())

    color = 'lightblue'
    if 'likes' in feature: color = 'green'
    if 'dislike' in feature: color = 'red'

    plt.subplot(2,2,idx+1)
    g = sns.distplot(df[feature], color=color)
    g.set_title(f'{feature} Distribution', fontsize=14)

plt.subplots_adjust(wspace=.2, hspace=.4, top=.9)
plt.show()


# In[ ]:


## log transform to get rid of highly skewed distribution
## log transformation and obtained "normal" distribution allows us to better analyze correlation
for feature in num_feat:
    df[feature+'_log'] = np.log(df[feature]+1)
    
num_log_feat = [f for f in df.columns if '_log' in f]


# In[ ]:


sns.set_style('dark')
plt.figure(figsize=(14,6))

for idx,feature in enumerate(num_log_feat):
    color = 'lightblue'
    if 'likes' in feature: color = 'green'
    if 'dislike' in feature: color = 'red'

    plt.subplot(2,2,idx+1)
    g = sns.distplot(df[feature], color=color)
    g.set_title(f'{feature} Distribution', fontsize=14)

plt.subplots_adjust(wspace=.2, hspace=.4, top=.9)
plt.show()


# In[ ]:


## check correlation matrix for numerical variables
plt.figure(figsize=(14,9))
sns.heatmap(df[num_log_feat].corr(), linewidths=.5, annot=True, cbar=False, cmap='Blues');


# By applying log transformation to all four numerical variables we were able to create a correlation matrix and plot it via heatmap:
# * the number of views seem to be highly correlated with likes, dislikes and the amount of comments
# * all the variables seem to depend on each other, the more views, the more engagement and vice versa
# 
# Let's take a look at the distribution of views per category and which category has the most videos

# # Number of Videos per Category & View Distribution

# In[ ]:


print('Top 5 Category Count:')
print(df['category'].value_counts()[:5])


# In[ ]:


plt.figure(figsize=(18,24))

plt.subplot(5,1,1)
g1 = sns.countplot(df['category'], data=df, palette='pastel')
g1.set_xticklabels(g.get_xticklabels(), rotation=45)
g1.set_xlabel('',fontsize=12)
g1.set_ylabel('Count', fontsize=12)
g1.set_title('No. of Videos per Category', fontsize=12, weight='bold')

for idx,feature in enumerate(num_log_feat):
    plt.subplot(5,1,idx+2)
    g = sns.boxplot(x='category', y=feature, data=df, palette='GnBu_d')
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_xlabel('', fontsize=12)
    g.set_ylabel(feature, fontsize=12)
    g.set_title(f'{feature} Distribution by Category', fontsize=12, weight='bold')

plt.subplots_adjust(hspace=0.6)
plt.show()


# after plotting a general overview per category we also created some boxplots to show the distribution of our log transformed data per category:
# * Entertainment and Music are the most prominent categories with the highest count of videos
# * Music, Film&Animation and Gaming get the most views, however the view distribution is quite equally spread over all present categories
# * nearly the same holds true for the like, dislike and comment distribution
# 
# In the next step, we will try to gain some more insights by creating new features from the publishing time variable

# # Publishing Time Features

# In[ ]:


## convert publish time into datetime
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['publish_year'] = df['publish_time'].dt.year
df['publish_month'] = df['publish_time'].dt.month
df['publish_day'] = df['publish_time'].dt.day_name()
df['publish_hour'] = df['publish_time'].dt.hour


# In[ ]:


## datetime features
publish_feat = [f for f in df.columns if 'publish' in f and not 'publish_time' in f]

plt.figure(figsize=(18,24))
rows = len(publish_feat)

for idx,feature in enumerate(publish_feat):
    plt.subplot(rows,1,idx+1)
    g = sns.boxplot(x=feature, y='views_log', data=df)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_xlabel('', fontsize=12)
    g.set_ylabel('views(log)', fontsize=12)
    g.set_title(f'Views per {feature}', fontsize=12, weight='bold')

plt.subplots_adjust(hspace=0.9)
plt.show()


# by looking at our new features we may come to the following conclusions:
# * Youtube's popularity seems to have risen since 2017, due to the general rise in views around 2017-2018
# * Publishing new videos around the summer month seems to be a bad idea, since the views are quite low - seems to be that Youtube is not an holiday activity
# * Publishing on a special day or hour seems to irrelevant, since the views are nearly equally distributed (Sunday might be the best day, however)
# 
# Let's take a look at which category gets the most engagement in terms of likes, dislikes and comments

# # Engagement Features

# In[ ]:


## create features based on the numerical variables in relation to the overall views
eng_feat = df[['likes','dislikes','comment_count']]

for feature in eng_feat:
    df[feature+'_rate'] = df[feature] / df['views'] * 100


# In[ ]:


eng_rate_feat = [f for f in df.columns if '_rate' in f]
rows = len(eng_rate_feat)

plt.figure(figsize=(12,18))

for idx, feature in enumerate(eng_rate_feat):
    plt.subplot(rows, 1, idx + 1)
    g = sns.boxplot(x='category', y=feature, data=df)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_xlabel('', fontsize=12)
    g.set_ylabel(feature, fontsize=12)
    g.set_title(f'{feature} by Category', fontsize=12, weight='bold')

plt.subplots_adjust(hspace=0.5)
plt.show()


# by depicting the engagement rates one might come to the following conclusions:
# * each category has a lot of outliers or high variance in terms of engagement rate
# * however, Music and Comedy seem to have a slight advantage when it comes to likes

# # General Conclusion and Outlook

# After completing the EDA for the US-Youtube dataset these are my final conclusions:
# * Views on youtube are highly skewed, which means there are just a few videos super successfull while the most videos are averaging around 0-700K views at most
# * Most prominent categories are Music, Entertainment and HowTo & Style
# * Youtube gained more traction or popularity after 2017
# * Publishing new videos around summer might not be the best idea, probably due to holiday season
# * a special day or daytime to post new content is not clearly visible
# 
# Since this was my very first notebook I would really appreciate some feedback, especially regarding the derived conclusions.
# In the future I would like to revisit this notebook and include some text-features and the correlation between certain "Buzz Words" and the view count.
# 
# Bye and thanks alot :)
