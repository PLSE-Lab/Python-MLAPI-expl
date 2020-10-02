#!/usr/bin/env python
# coding: utf-8

# # Avito Demand Prediction - Data Exploration
# *Basis notebook covering the training and test data based on distributions, time series and relationships with the target variable.*

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from dataninja import kick
import altair as alt
import seaborn as sns
import cufflinks as cf
import plotly.offline as plotly
get_ipython().run_line_magic('matplotlib', 'inline')

cf.go_offline()


# ### Import Data

# In[3]:


get_ipython().run_cell_magic('time', '', 'dtypes = {\'user_type\':\'category\',\n          \'category_name\':\'category\',\n          \'parent_category_name\':\'category\',\n          \'region\':\'category\',\n          \'city\':\'category\'}\n\ntrain = pd.read_csv("../input/train.csv",\n                    parse_dates=[\'activation_date\'],\n                    dtype=dtypes)\ntest = pd.read_csv("../input/test.csv",\n                   parse_dates=[\'activation_date\'],\n                   dtype=dtypes)\n\ntarget = \'deal_probability\'')


# In[4]:


all_data = pd.concat([train.assign(dataset='train'),test.assign(dataset='test')],sort=False).reset_index(drop=True)


# ### Exploration

# In[5]:


dataset_colors = ['red','orange']


# #### Unique Values

# In[6]:


(all_data
 .groupby('dataset')
 .nunique()
 .unstack()
 .sort_values(ascending=False)
 .unstack()
 .sort_values('test',ascending=False)
 .iloc[8:]
 .drop([target]+['dataset'])
 .iplot(title='Unique Value Counts for Categoricals by Dataset',kind='bar',colors=dataset_colors)
)


# #### Missing Values 

# In[7]:


#features with missing values
missing_vals = all_data.isnull().sum()
missing_vals = missing_vals[missing_vals>0]
missing_vals = missing_vals.index.tolist()
missing_vals.remove(target)

obs_counts = all_data.groupby('dataset').size().reset_index(name='obs_count')

(all_data
 .groupby('dataset')
 .apply(lambda x: x.isnull().sum())
 .merge(obs_counts,left_index=True,right_on='dataset')
 .set_index('dataset')
 .loc[:,missing_vals+['obs_count']]
 .transform(lambda x: x/x.max(),axis=1)
 .drop('obs_count',axis=1)
 .T
 .iplot(kind='bar',title='Missing Values Comparison - Train vs. Test',colors=dataset_colors)
)


# In[8]:


(all_data
 .set_index('dataset')
 .loc[:,missing_vals]
 .isnull().sum(axis=1)
 .reset_index(name='item_missing_val_count')
 .groupby(['dataset','item_missing_val_count'])
 .size().reset_index(name='item_missing_val_count_count')
 .merge(obs_counts,on='dataset')
 .assign(item_missing_val_count_scaled = lambda x: x.item_missing_val_count_count/x.obs_count)
 .set_index(['dataset','item_missing_val_count']).item_missing_val_count_scaled
 .unstack(0)
 .iplot(title='Missing Values per Item Comparisons',kind='bar',colors=dataset_colors)
)


# #### Target Variable Averages by Features

# In[9]:


train.groupby('user_type')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by User Type',color='blue')


# In[10]:


train.groupby('parent_category_name')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Parent Category',color='green')


# In[11]:


train.groupby('category_name')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Category',color='gold',margin={'b':120})


# In[12]:


train.groupby('region')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Region',color='magenta')


# In[13]:


train.query('price<5000000').groupby('price')[target].mean().rolling(1000,min_periods=50).mean().iplot(title='Deal Probabilty by Price',color='purple')


# In[14]:


train.groupby('image_top_1')[target].mean().rolling(1000,min_periods=50).mean().iplot(title='Deal Probabilty by Image Top 1',color='black')


# #### Time Series Analysis

# In[15]:


all_data.groupby(['dataset','activation_date']).size().unstack(0).iplot(title='Train vs. Test by Activation Date',colors=dataset_colors)


# In[16]:


fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(data = all_data
            .assign(weekday = lambda x: x.activation_date.dt.weekday,
                    week = lambda x: x.activation_date.dt.week)
            .groupby(['weekday','week'])
            .size()
            .unstack(0),
            cmap='viridis',
            ax=ax
            )
plt.title('Listings by Week and Day');


# #### Feature Distribution

# In[17]:


all_data.description.str.len().hist(bins=50,color='black',figsize=(10,6))
plt.title('Description Length Distribution');


# In[18]:


all_data.title.str.len().hist(bins=50,color='gold',figsize=(10,6))
plt.title('Title Length Distribution');


# #### Target Distribution 

# In[19]:


train[target].iplot(kind='hist',bins=20,title='Deal Probability Distribution')


# In[20]:


train[target].value_counts().transform(lambda x: x/x.sum()).head(10).round(2)


# Deal probabilities **potentially created with a non-continuous model such as a GBM** as many listings share duplicate deal probabilities indicating they **likely belong in the same prediction bin**.
