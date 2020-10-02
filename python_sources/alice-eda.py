#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[ ]:


# Import libraries and set desired options
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from tqdm import tqdm
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# reading from csv
train_df = pd.read_csv('data/train_sessions.csv',
                       index_col='session_id', parse_dates=['time1'])
test_df = pd.read_csv('data/test_sessions.csv',
                      index_col='session_id', parse_dates=['time1'])

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head(3)


# In[ ]:


sites = ['site%s' % i for i in range(1, 11)]
times = ['time%s' % i for i in range(1, 11)]


# In[ ]:


train_df['year']=train_df['time1'].apply(lambda arr:arr.year)
train_df['hour']=train_df['time1'].apply(lambda arr:arr.hour)
train_df['day_of_week']=train_df['time1'].apply(lambda t: t.weekday())
train_df['month']=train_df['time1'].apply(lambda t: t.month)
sessduration = (train_df[times].apply(pd.to_datetime).max(axis=1) - train_df[times].apply(pd.to_datetime).min(axis=1)).astype('timedelta64[ms]').astype('int')
train_df['sessduration']=np.log1p(sessduration)
#train_df['sessduration'] = sessduration
#train_df['sessduration']=StandardScaler().fit_transform(sessduration.values.reshape(-1, 1))


# In[ ]:


train_df[train_df["target"]==1].head(5)


# In[ ]:


train_df.fillna(0).info()


# In[ ]:


train_df.describe()


# In[ ]:


# How many times it was Alice
train_df['target'].value_counts(normalize=False)


# In[ ]:


train_df.groupby('target').count()


# In[ ]:


train_df.head(5)


# In[ ]:


train_df.groupby('target')['sessduration','month','day_of_week','hour','year'].describe().T


# # Visualization

# In[ ]:


train_df.fillna(0).tail(3)


# ## Quantitative

# In[ ]:


plt.figure(figsize=(10, 4))
sns.distplot(train_df['sessduration'])


# In[ ]:


plt.figure(figsize=(10, 4))
sns.distplot(train_df[train_df['target']==1]['sessduration'])


# ## Categorical

# In[ ]:


train_df.columns


# In[ ]:


sns.countplot(x='day_of_week',data=train_df[train_df['target']==0])


# In[ ]:


sns.countplot(x='day_of_week',data=train_df[train_df['target']==1])


# In[ ]:


sns.countplot(x='hour',data=train_df[train_df['target']==0])


# In[ ]:


sns.countplot(x='hour',data=train_df[train_df['target']==1])


# In[ ]:


sns.countplot(x='year',data=train_df[train_df['target']==0])


# In[ ]:


sns.countplot(x='year',data=train_df[train_df['target']==1])


# In[ ]:


sns.countplot(x='month',data=train_df[train_df['target']==0])


# In[ ]:


sns.countplot(x='month',data=train_df[train_df['target']==1])


# ## Multivariate

# In[ ]:


corr_matrix = train_df[['hour','day_of_week', 'month', 'sessduration']].corr()
sns.heatmap(corr_matrix,cmap='coolwarm');


# In[ ]:


plt.figure(figsize=(8, 4))
sns.boxplot(x='target',y='sessduration',data=train_df)


# In[ ]:


plt.figure(figsize=(8, 4))
sns.boxplot(x='target',y='hour',data=train_df)


# In[ ]:


plt.figure(figsize=(8, 4))
sns.boxplot(x='target',y='day_of_week',data=train_df)

