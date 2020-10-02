#!/usr/bin/env python
# coding: utf-8

# # Objective
# [Heads or Tails](https://www.kaggle.com/headsortails/shopping-for-insights-favorita-eda) made a wonderful EDA in R two months ago and analyzed almost every aspects of the data  in Favorita competetion. No wonder he got more than 250 upvote with the kernel. For a beginner in Kaggle and a late paticipator in Favorita challenge, I felt astonished when I saw his beautiful plots and detailed analysis. I definitly want to go to the daRk side to enjoy all the fantastic features from ggplot2 library. However, since I already chose Python as my primary language in ML research, I just want to absorb all his ideas and skills in data analysis and visualization. What I can do is to reproduce his work in Python and practice my coding skills through this process. I found there are still many minor features I don't know how to express in the plots. Hope you can share your comments and methods after reading my kernel. Thanks.

# ## 1. Loading data

# In[ ]:


import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import lightgbm as lgb
from xgboost import XGBRegressor
import xgboost
from time import time
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
import random
import time


# In[ ]:


from subprocess import check_output
print (check_output(['ls', '../input']).decode('utf8'))


# The `train` dataset include more than 125 million lines. It takes forever to analyze them and plot. So I will only analyze 5% of them. 

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train_sample = train.sample(frac = 0.05)
train_sample.head()


# In[ ]:


holiday_events = pd.read_csv('../input/holidays_events.csv')
items = pd.read_csv('../input/items.csv')
oil = pd.read_csv('../input/oil.csv')
stores = pd.read_csv('../input/stores.csv')
test = pd.read_csv('../input/test.csv')

transactions = pd.read_csv('../input/transactions.csv')


# In[ ]:


print (holiday_events.head())
print (holiday_events.shape)
print (holiday_events.describe())


# In[ ]:


print (items.head())
print (items.shape)
print (items.describe())


# In[ ]:


print (len(items['family'].value_counts().index))
print (len(items['class'].value_counts().index))
print (items['perishable'].value_counts())


# From `item` dataset, we can find:
#     1. There are 4100 items.
#     2. These items can be classfied to 33 families (e.g. GROCERY I)
#     3. There are 337 classes
#     4. 3114 of them are not perishable, 986 of them are perishable

# In[ ]:


print (oil.head())
print (oil.shape)
print (oil.describe())


# In[ ]:


import datetime
train_sample['date'] = train_sample['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))
train_sample.sort_values(by = 'date', inplace = True)
print (train_sample.head())
print (train_sample.shape)
print (train_sample.describe())
print (train_sample.isnull().sum())


# In[ ]:


train_sample['onpromotion'].fillna(value = 'missing', inplace = True)
#train_sample['onpromotion'].isnull().sum()
print (train_sample['onpromotion'].value_counts())


# In[ ]:


print (test.head())
print (test.shape)
print (test.describe())
print (test.isnull().sum())


# In[ ]:


print (test['onpromotion'].value_counts())


# # 2. Individual feature visualisations
# 
# In this step we will get an visual overview of the data by plotting the individual feature distribution for each data set sepeartely.

# ## 3. Training data

# In[ ]:


daily_sale = pd.DataFrame()
daily_sale['count'] = train_sample['date'].value_counts()
daily_sale['date'] = train_sample['date'].value_counts().index
daily_sale = daily_sale.sort_values(by = 'date')
print (daily_sale.head(3))
unit_sale = pd.DataFrame()
unit_sale['count'] = train_sample[train['unit_sales']>0]['unit_sales'].value_counts()
unit_sale['positive_unit_sales'] = train_sample[train['unit_sales']>0]['unit_sales'].value_counts().index
unit_sale = unit_sale.sort_values(by = 'positive_unit_sales')


# In[ ]:


promotion_count = pd.DataFrame()
promotion_count ['count'] = train_sample['onpromotion'].value_counts()


# In[ ]:


fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (18,5))
sns.set_style('darkgrid')
ax1.plot(daily_sale['date'], daily_sale['count'])
ax1.set_xlabel('date', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelsize=15)
sns.barplot(x = promotion_count.index, y = promotion_count['count'],ax = ax2)
ax2.set_ylabel('count', fontsize = 15)
ax2.set_xlabel ('onpromotion',fontsize =15)
ax2.tick_params(labelsize = 15)
ax2.ticklabel_format(style = 'sci',scilimits = (0,0), axis = 'y')
plt.subplot(1,3,3)
plt.loglog(unit_sale['positive_unit_sales'], unit_sale['count'])
plt.ylabel('count', fontsize = 15)
plt.xlabel('positive_unit_sales', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()


# In[ ]:


store_count = pd.DataFrame()
store_count['count'] = train_sample['store_nbr'].value_counts().sort_index()


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 3))
sns.barplot(x = store_count.index, y = store_count['count'], ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('store_nbr',fontsize = 15)
ax.tick_params(labelsize=15)


# In[ ]:


item_count = pd.DataFrame()
item_count['count'] = train_sample['item_nbr'].value_counts().sort_index()
plt.plot(item_count.index)


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 3))
sns.barplot(x = item_count.index, y = item_count['count'], ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('item_nbr',fontsize = 15)
ax.tick_params(axis = 'x',which = 'both',top = 'off', bottom = 'off', labelbottom = 'off')


# In[ ]:


neg_unit_sale = pd.DataFrame()
neg_unit_sale['unit_sales'] = (-train_sample[train_sample['unit_sales'] < 0]['unit_sales'])
neg_unit_sale.head()


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 5))
#ax.set_xscale('log')
np.log(neg_unit_sale['unit_sales']).plot.hist(ax = ax, log = True,edgecolor = 'white', bins = 50)
ax.set_xlabel('neg_unit_sales (log10 scale)', fontsize=15)
ax.set_ylabel('count', fontsize=15)
ax.tick_params(labelsize=15)


# ## 2.2 Stores

# In[ ]:


city_count = pd.DataFrame()
city_count['count'] = stores['city'].value_counts().sort_index()
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18, 4))
g = sns.barplot(x = city_count.index, y = city_count['count'], ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('city',fontsize = 15)
ax1.tick_params(labelsize=15)
g.set_xticklabels(city_count.index, rotation = 45)
state_count =pd.DataFrame()
state_count['count'] = stores['state'].value_counts().sort_index()
g2 = sns.barplot(x = state_count.index, y = state_count['count'], ax = ax2)
ax2.set_ylabel('count', fontsize = 15)
ax2.set_xlabel('state',fontsize = 15)
ax2.tick_params(labelsize=15)
g2.set_xticklabels(state_count.index, rotation = 45)


# In[ ]:


type_count = pd.DataFrame()
type_count['count'] = stores['type'].value_counts().sort_index()
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18, 4))
g = sns.barplot(x = type_count.index, y = type_count['count'], ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('type',fontsize = 15)
ax.tick_params(labelsize=15)
cluster_count = pd.DataFrame()
cluster_count['count'] = stores['cluster'].value_counts().sort_index()
g = sns.barplot(x = cluster_count.index, y = cluster_count['count'], ax = ax2)
ax2.set_ylabel('count', fontsize = 15)
ax2.set_xlabel('cluster',fontsize = 15)
ax2.tick_params(labelsize=15)


# In[ ]:


import squarify


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
grouped_city = np.log1p(stores.groupby(['city']).count())
grouped_city['state'] = grouped_city.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_city['store_nbr'], label = grouped_city.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 25)
plt.axis('off')


# ### For the treemap, I used squarify to show the `count` as areas of different `city` and `state`. I don't know how to plot sub treemap in a state square to illustrate  `city` count. If you know how to do it, please make a comment

# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
grouped_state = np.log1p(stores.groupby(['state']).count())
grouped_state['state'] = grouped_state.index
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_state['store_nbr'], label = grouped_state.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 20)
plt.axis('off')


# In[ ]:


print (items.head())
class_count = pd.DataFrame()
class_count['count'] = items['class'].value_counts().sort_index()
most_freq_class = items['class'].value_counts().head()
perish_count = items['perishable'].value_counts()
family_count = items['family'].value_counts().sort_index()


# ## 2.3 Items

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (18,3))
sns.barplot(x = class_count.index, y = class_count['count'], ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('class',fontsize = 15)
ax1.tick_params(labelsize = 6, axis = 'x',which = 'both',top = 'off', bottom = 'off', labelbottom = 'off')
sns.barplot(x = most_freq_class.index, y = most_freq_class.values, ax = ax2)
ax2.set_ylabel('n', fontsize = 15)
ax2.set_xlabel('class-most frequent',fontsize = 15)
ax2.tick_params(labelsize = 12)
sns.barplot(x = perish_count.index, y = perish_count.values, ax = ax3)
ax3.set_ylabel('count', fontsize = 15)
ax3.set_xlabel('perish',fontsize = 15)
ax3.tick_params(labelsize = 12)


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 3))
g = sns.barplot(x = family_count.index, y = family_count.values, ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('family',fontsize = 15)
ax.tick_params(labelsize = 12)
g.set_xticklabels(family_count.index, rotation = 60)


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
grouped_family = np.log1p(items.groupby(['family']).count().head(20))
current_palette = sns.color_palette()
squarify.plot(sizes = grouped_family['class'], label = grouped_family.index, alpha = 0.8,color = current_palette)
plt.rc('font', size = 12)
plt.axis('off')


# In[ ]:


med_trans = transactions.groupby(['date']).median()
med_trans['date'] = med_trans.index
med_trans['date'] = med_trans['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))
med_trans.head()


# ## 2.4 Transactions
# 
# I can reproduce Head and Tails' med_trans vs. date plot but the feature I missed is the trend curve. It seems very convinient and easy to fulfill in `R` with `geom_smooth` but I don't know how to do it in Python. Please let me know in the comment if you have a elegant solution. 

# In[ ]:


fig, ax = plt.subplots(figsize = (12,6))
sns.set_style('darkgrid')
plt.plot(med_trans['date'], med_trans['transactions'])
#ax = sns.regplot(x='date', y='transactions', data = med_trans)
ax.set_xlabel('date', fontsize=15)
ax.set_ylabel('med_trans', fontsize=15)
ax.tick_params(labelsize=15)


# Since I don't know how to make smooth trend curve of time series plot, I don't have acess to the smoothed transations time series for each individual store. 

# In[ ]:


transactions['date'] = transactions['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))
grouped_trans = transactions.groupby(['date','store_nbr']).sum().unstack()
grouped_trans.head()
fig, ax = plt.subplots(figsize = (12,6))
grouped_trans.plot(ax = ax)


# 

# ## 2.5 Oil
# 
# Below is the plot of price over time together with its weekly changes (price - price 7 days later):

# In[ ]:


oil['date'] = oil['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))


# In[ ]:


import statsmodels.formula.api as sm
regression = (sm.ols(formula="dcoilwtico ~ date ", data=oil).fit())


# In[ ]:


oil['trend'] = regression.predict(oil['date'])


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 5))
ax.plot(oil['date'],oil['dcoilwtico'], label = 'oilprice')
ax.set
ax.set_xlabel('date', fontsize=15)
ax.set_ylabel('oilprice', fontsize=15)
ax.tick_params(labelsize=15)


# In[ ]:


lag7 = oil['dcoilwtico'] - oil['dcoilwtico'].shift(7)


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 5))
ax.plot(oil['date'],lag7, label = 'weekly variations in oil price')
ax.set
ax.set_xlabel('date', fontsize=15)
ax.set_ylabel('weekly variations', fontsize=15)
ax.tick_params(labelsize=15)


# ## 2.6 Holidays

# In[ ]:


holiday_events.head()


# In[ ]:


holiday_events['date'] = holiday_events['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))


# In[ ]:


type_count = holiday_events['type'].value_counts().sort_index()
locale_count = holiday_events['locale'].value_counts().sort_index()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18, 4))
sns.barplot(x = type_count.index, y = type_count.values, ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('type',fontsize = 15)
sns.barplot(x = locale_count.index, y = locale_count.values, ax = ax2)
ax2.set_ylabel('count', fontsize = 15)
ax2.set_xlabel('locale',fontsize = 15)


# In[ ]:


most_freq_descr = holiday_events['description'].value_counts().head(12).sort_index()
#print (most_freq_descr)
transferred_count = holiday_events['transferred'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18, 4))
sns.barplot(y = most_freq_descr.index, x = most_freq_descr.values, ax = ax1)
ax1.set_ylabel('Description-most frequent', fontsize = 15)
ax1.set_xlabel('Frequency',fontsize = 15)
sns.barplot(x = transferred_count.index, y = transferred_count.values, ax = ax2)
ax2.set_ylabel('count', fontsize = 15)
ax2.set_xlabel('transferred',fontsize = 15)


# In[ ]:


locale_name_count = holiday_events['locale_name'].value_counts().sort_index()
locale_name_count.head()
fig, ax = plt.subplots(figsize = (18, 3))
g = sns.barplot(x = locale_name_count.index, y = locale_name_count.values, ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('locale_name',fontsize = 15)
ax.tick_params(labelsize = 12)
g.set_xticklabels(locale_name_count.index, rotation = 45)


# ## Summary
# 
# I have completed the basic visualisation of datasets in the project. The following will be the analysis and plots combining different datasets together. The advantage of `ggplot` library in `R` is very obvious. It's easy to add legend, superposition of different plot, get smoothed trend curve and make plots pretty. 

# In[ ]:




