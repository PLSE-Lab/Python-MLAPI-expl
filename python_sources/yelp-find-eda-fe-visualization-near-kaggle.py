#!/usr/bin/env python
# coding: utf-8

# <a id="intro">

# # Yelp: Find EDA, FE, Visualization near Kaggle
# ![](https://s3-media3.fl.yelpcdn.com/assets/srv0/seo_metadata/f9149736ad8d/assets/img/logos/yelp_og_image.png)
# 
# ### Context
# This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In the dataset you'll find information about businesses across 11 metropolitan areas in four countries.
# 
# ### Content
# This dataset contains seven CSV files and the original JSON files can be found in yelp_academic_dataset.zip. 
# 
# __In total, there are__ :
# - __5,200,000__ user reviews
# - Information on __174,000__ businesses
# - The data spans __11__ metropolitan areas
# 
# ### Inspiration
# Natural Language Processing & Sentiment Analysis
# What's in a review? Is it positive or negative? Yelp's reviews contain a lot of metadata that can be mined and used to infer meaning, business attributes, and sentiment.
# 
# __Graph Mining__
# 
# We recently launched our Local Graph but can you take the graph further? How do user's relationships define their usage patterns? Where are the trend setters eating before it becomes popular?
# 
# ### Author's word
# Hello everyone. I'm glad to invite you to this Yelp travel. We'll try to find EDA, FE, Visualization near Kaggle location. Python will help us in this hard journey.
# 
# ### Table of contents:
# 1. [Introduction](#intro)
# 2. [Very first data exploration (EDA)](#eda1)
#     - [Closer view to user](#eda1_user)
#     - [Closer view to business](#eda1_business)
#     - [Closer view to review](#eda1_review)
#     - [Closer view to tip](#eda1_tip)
#     - [Closer view to business_hours](#eda1_bh)
#     - [Closer view to checkin](#eda1_checkin)
#     - [Closer view to business_attributes](#eda1_ba)
# 3. [Data preparation and feature engineering (FE)](#dataprep1)
# 5. [Data visualization](#dv1)
# 6. [Summary](#summary)
# 

# In[ ]:


## imports and settings
import os
import re
import math
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot, iplot, init_notebook_mode
import networkx as nx
import warnings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
warnings.simplefilter('ignore')
init_notebook_mode()

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id="eda1">
# ### 2. Very first data exploration

# In[ ]:


## Explore input foler
os.listdir('../input')


# In[ ]:


get_ipython().run_cell_magic('time', '', "## Load data\nuser = pd.read_csv('../input/yelp_user.csv')\nbusiness = pd.read_csv('../input/yelp_business.csv')\nbusiness_hours = pd.read_csv('../input/yelp_business_hours.csv', na_values='None')\nbusiness_attributes = pd.read_csv('../input/yelp_business_attributes.csv')\ncheckin = pd.read_csv('../input/yelp_checkin.csv')\ntip = pd.read_csv('../input/yelp_tip.csv')\nreview = pd.read_csv('../input/yelp_review.csv')\n\n## Little types transform\nuser['yelping_since'] = pd.to_datetime(user['yelping_since'])")


# In[ ]:


## Define dataset names
dnames = ['user', 'business', 'business_hours', 'business_attributes', 'checkin', 'tip', 'review']
## Explore shapes of datasets
for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):
    print(n, d.shape)


# In[ ]:


## First look at datasets
for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):
    ## business_attrubutes is too many columns
    print('---------{0}---------'.format(n))
    if n != 'business_attributes':
        print(d.head(1).T)
    else:
        print(d.columns)


# In[ ]:


## Find keys in datasets
colnames = []
for d in [user, business, business_hours, business_attributes, checkin, tip, review]:
    colnames.extend(d.columns)
colnames = pd.Series(colnames).value_counts().reset_index()
colnames.columns = ['colname', 'cnt']
colnames[colnames['cnt'] > 1]


# __Keys are:__
# - __business_id__- organization identifier
# - __user_id__ - user identifier

# In[ ]:


G = nx.Graph()
fig, ax = plt.subplots(figsize=[7,7])
for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):
    _ = []
    for c in np.intersect1d(d.columns, ['business_id', 'user_id']):
        _.append([n, c])
    G.add_edges_from(_, label=n)
nx.draw_networkx(G, ax=ax)
plt.show()


# #### Short summary:
# - __users__ (_user_id_) make __reviews__ and __tips__
# - every __review__ and __tip__  point out __business__ (_buisiness_id_)
# - __buisiness__ is Organization, it's profile is represented in __buisiness_hours__, __buisiness_attributes__, __checkin__

# In[ ]:


## How many users make reviews?
print('Total users:', user['user_id'].nunique())
## How many user_id's contains in reviews
print('Total users review:', review['user_id'].nunique())
## Is there any different user_id?
print('Different user_id:', np.setdiff1d(review['user_id'], user['user_id']),
      review['user_id'].nunique()/user['user_id'].nunique())
## How many observations of this different user_id?
print('Different user_id shape:', 
      review[review['user_id'].isin(np.setdiff1d(review['user_id'], user['user_id']))].shape)
## How many user_id's contains in tip
print('Total users tip:', tip['user_id'].nunique())
## Is there any different user_id?
print('Tip users - users:', len(np.setdiff1d(tip['user_id'], user['user_id'])),
      tip['user_id'].nunique()/user['user_id'].nunique())
print('Users - tip users:', len(np.setdiff1d(user['user_id'], tip['user_id'])))


# #### Short summary
# - We have total __1 326 100__ users
# - All users make reviews
# - One user in wide review dataset but not in user stats dataset
# - Only __20% (271679)__ of users make tips

# In[ ]:


## How many organization contains our dataset?
print('Total organization:', business['business_id'].nunique())
## How many organizations has reviews?
print('Total organizations with review:', review['business_id'].nunique(),
      review['business_id'].nunique()/business['business_id'].nunique())
## How many organizations has tip?
print('Total organization with tip:', tip['business_id'].nunique(),
      tip['business_id'].nunique()/business['business_id'].nunique())
## How many organization have buisiness_hours info?
print('Total organizations with buisiness hours info:', business_hours['business_id'].nunique(),
      business_hours['business_id'].nunique()/business['business_id'].nunique())
## How many organization have buisiness_attributes info?
print('Total organizations with buisiness attributes info:',business_attributes['business_id'].nunique(),
      business_attributes['business_id'].nunique()/business['business_id'].nunique())
## How many organization have checkin info?
print('Total organizations with checkin info:', checkin['business_id'].nunique(),
      checkin['business_id'].nunique()/business['business_id'].nunique())


# #### Short summary:
# - We have total __174 567__ organizations
# - All organizations has reviews
# - Only __64% (112365)__ organizations has tips
# - All organizations has business hours info
# - Only __87% (152 041)__ of organizations has business attrubutes
# - Only __83% (146 350)__ of organizations has checkin info
# 

# <a id="eda1_user">
# ### Closer view to user

# In[ ]:


## Look at the most active users
user.sort_values('review_count', ascending=False).head(2).T


# In[ ]:


user.describe(include='all').T


# In[ ]:


#### Look at distribution for numeric variables
user_desc = user.dtypes.reset_index()
user_desc.columns = ['variable', 'type']
cols = user_desc[user_desc['type']=='int64']['variable']
fig, ax = plt.subplots(math.ceil(len(cols)/2), 2, figsize=[12, math.ceil(len(cols)/2)*2])
ax = list(itertools.chain.from_iterable(ax))
for ax_, v in zip(ax[:len(cols)], cols):
    sns.distplot(np.log1p(user[v]), ax=ax_, label=v)
    ax_.set_xticklabels(np.expm1(ax_.get_xticks()).round())
    ax_.legend()
plt.show()


# <a id="eda1_business">
# ### Closer view to business

# In[ ]:


business.head(2).T


# In[ ]:


business.describe(include='all').T


# In[ ]:


#### Look at distribution for numeric variables
business_desc = business.dtypes.reset_index()
business_desc.columns = ['variable', 'type']
cols = business_desc[business_desc['type']=='int64']['variable']
fig, ax = plt.subplots(math.ceil(len(cols)/2), 2, figsize=[12, math.ceil(len(cols)/2)*2])
for ax_, v in zip(ax[:len(cols)], cols):
    sns.distplot(np.log1p(business[v]), ax=ax_, label=v)
    ax_.set_xticklabels(np.expm1(ax_.get_xticks()).round())
    ax_.legend()
plt.show()


# In[ ]:


## Look at count for categorical variables
cols = ['neighborhood', 'city', 'state']
for c in cols:
    print(business[c].value_counts(normalize=True).head())


# In[ ]:


## What about categories of organizations
## How many categories in each organization? (minuimum 1, maximum 35 categories)
## Most frequent 2 categories
print(business['categories'].str.count(';').min() + 1, business['categories'].str.count(';').max() + 1)
(business['categories'].str.count(';') + 1).value_counts().head()


# In[ ]:


## How many categories we have?
categories = pd.concat(
    [pd.Series(row['business_id'], row['categories'].split(';')) for _, row in business.iterrows()]
).reset_index()
categories.columns = ['categorie', 'business_id']
categories.head()


# In[ ]:


## How many categories?
print(categories['categorie'].nunique())
## Most frequent categories
categories['categorie'].value_counts().head(10)


# In[ ]:


fig, ax = plt.subplots(figsize=[5,10])
sns.countplot(data=categories[categories['categorie'].isin(
    categories['categorie'].value_counts().head(25).index)],
                              y='categorie', ax=ax)
plt.show()


# In[ ]:


categories_ = categories[
    (categories['categorie'].isin(categories['categorie'].value_counts().head(25).index))
]
ct = pd.crosstab(
    categories_['business_id'],
    categories_['categorie'])

fig, ax = plt.subplots(figsize=[10,10])
sns.heatmap(ct.head(25), ax=ax, cmap='Reds')
ax.set_title('Top 25-cat, Random 25 organizations')


# In[ ]:


## Also we have geospatial data
g = sns.jointplot(data=business, x='longitude', y='latitude', size=8, stat_func=None)


# <a id="eda1_review">
# ### Closer view to review

# In[ ]:


review.head(2).T


# In[ ]:


review.describe(include='all').T


# #### Short summary to review dataset:
# - __user__ dataset represent aggregate information from __review__
# - in __review__ dataset we have uncompressed text reviews - This might be useful in case of sentimental analysis.

# <a id="eda1_tip">
# ### Closer view on tip

# In[ ]:


tip.head(2).T


# In[ ]:


tip.describe(include='all').T


# <a id="eda1_bh">
# ### Closer view to business_hour

# In[ ]:


business_hours.head()


# In[ ]:


business_hours.describe(include='all').T


# <a id="eda1_checkin">
# ### Closer view to checkin

# In[ ]:


checkin.head()


# In[ ]:


checkin.describe(include='all').T


# <a id="eda1_ba">
# ### Closer view to business_attribues

# In[ ]:


business_attributes.columns


# <a id="dataprep1">
# ### 3. Data preparation and feature engineering

# #### What we should do in data preparation part:
# - Calculate extreme values for __working days/holidays__ for every __business_id__
# - Join prepared __business_hours __ data with __business__ -> __business_ __ dataframe
# - Join __checkin__ with __business__ -> __business_ __ dataframe

# In[ ]:


## function for get time_range from string
def get_time_range(s):
    if isinstance(s, str):
        t1, t2 = s.split('-')
        h1, m1 = map(int, t1.split(':'))
        h2, m2 = map(int, t2.split(':'))
        m1, m2 = m1/60, m2/60
        t1, t2 = h1+m1, h2+m2
        if t2 < t1:
            d = t2+24-t1
        else:
            d = t2-t1
        return t1, t2, d
    else:
        return None, None, None


# In[ ]:


get_ipython().run_cell_magic('time', '', "## Prepare start/finish/delta features for every weekday\nbh_colnames = business_hours.columns\nfor c in bh_colnames[1:]:\n    business_hours['{0}_s'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[0])\n    business_hours['{0}_f'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[1])\n    business_hours['{0}_d'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[2])\nbusiness_hours = business_hours.drop(bh_colnames[1:], axis=1)")


# In[ ]:


business_hours.head()


# In[ ]:


## Look at our features
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[15, 4])
sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_s' in c]].corr(),
            cmap='Reds', ax=ax1)
sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_f' in c]].corr(),
            cmap='Greens', ax=ax2)
sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_d' in c]].corr(),
            cmap='Blues', ax=ax3)
ax1.set_title('Start hours heatmap')
ax2.set_title('Finish hours heatmap')
ax3.set_title('Duration heatmap')
plt.show()


# In[ ]:


## Look at our features
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=[15, 9])
for wd in [c for c in business_hours.columns if '_s' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax1, label=wd)
for wd in [c for c in business_hours.columns if '_f' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax2, label=wd)
for wd in [c for c in business_hours.columns if '_d' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax3, label=wd)
ax1.legend()
ax2.legend()
ax3.legend()
ax1.set_title('Start hours distribution')
ax2.set_title('Finish hours distribution')
ax3.set_title('Duration distribution')
plt.show()


# #### Short summary
# - In general in working days from Monday to Friday - organizations start working in one time
# - In general from Friday to Saturday closing time extended
# - Working days working hours profile slightly differ from weekends
# 
# According to this conclusins we might group weekdays into 3 groups:
# - wd - Monday to Thursday
# - fr - Friday
# - ho - Saturday to Sunday
# 
# And culculate mean values for:
# - starting hours
# - finish hours
# - working duration

# In[ ]:


wd = ['mo', 'tu', 'we', 'th']
fr = ['fr']
ho = ['sa', 'su']

## define new_cols
bh_newcols = ['business_id']
for wg_name, wg in zip(['wd', 'fr', 'ho'], [wd, fr, ho]):
    for f in ['s', 'f', 'd']:
        cols = list(map(lambda d: '{0}_{1}'.format(d,f), wg))
        bh_newcols.append('{0}_{1}'.format(wg_name, f))
        business_hours['{0}_{1}'.format(wg_name, f)] = business_hours.loc[:, cols].median(axis=1)

business_hours.loc[:, bh_newcols].head()


# In[ ]:


## Look at our new features distribution
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=[15, 9])
for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_s' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax1, label=wd)
for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_f' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax2, label=wd)
for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_d' in c]:  
    sns.distplot(business_hours[wd].dropna(), ax=ax3, label=wd)
ax1.legend()
ax2.legend()
ax3.legend()
ax1.set_title('Start hours distribution')
ax2.set_title('Finish hours distribution')
ax3.set_title('Duration distribution')
plt.show()


# In[ ]:


## Join our new features to business dataframe
business = business.merge(business_hours.loc[:, bh_newcols])
business.head(2).T


# <a id="dv1">
# ### 4. Data visualization
# __need Add some fancy dataviz__

# ### Disclaimer
# This is not the final version of kernel. Plesase keep in touch.
