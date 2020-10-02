#!/usr/bin/env python
# coding: utf-8

# In[50]:


# some necessary imports
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 10)


# In[51]:


get_ipython().system('ls ../input')


# In[52]:


train_df = pd.read_csv('../input/train-balanced-sarcasm.csv')


# In[53]:


train_df.head()


# In[54]:


train_df.info()


# In[55]:


train_df.dropna(subset=['comment'], inplace=True)


# In[8]:


train_df.label.value_counts()


# In[9]:


train_comments, valid_comments, y_train, y_valid = train_test_split(train_df['comment'], train_df['label'], random_state=17)


# ## First, let's make an exploratory data analysis to detect some important features

# In[10]:


subreddits_to_plot = train_df.subreddit.value_counts().head(30).index


# In[11]:


plot = sns.countplot(x='subreddit', data=train_df[train_df.subreddit.isin(subreddits_to_plot)], hue='label')
_ = plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


# **We can clearly see, that the percentage of sarcastic comments differs from subreddit to subreddit**

# In[12]:


time_label_data = train_df.groupby(['date']).label.value_counts(normalize=True)


# In[13]:


plot = sns.lineplot(data=time_label_data.loc[:, 1])
for item in plot.get_xticklabels():
    item.set_rotation(90)


# **The general trend is that the amount of sarcasm in trands is decreasing**  
# ### Let's try to check if it's seasonal data

# In[14]:


# get new features from existing
train_df['year'] = train_df.date.apply(lambda x: x.split('-')[0])
train_df['month'] = train_df.date.apply(lambda x: x.split('-')[1])


# In[15]:


seasonal_time_year = train_df.groupby(['year', 'month']).label.value_counts(normalize=True).loc[:,:,1].reset_index(level=[0, 1])
seasonal_time_year.month = seasonal_time_year.month.astype(int)
seasonal_time_year.head()


# **Create 4 new seasonal features: autumn, winter, spring and summer**

# In[16]:


seasonal_time_year.loc[(seasonal_time_year.month >= 3) & (seasonal_time_year.month <= 5), 'season'] = 'spring'
seasonal_time_year.loc[(seasonal_time_year.month >= 6) & (seasonal_time_year.month <= 8), 'season'] = 'summer'
seasonal_time_year.loc[(seasonal_time_year.month >= 9) & (seasonal_time_year.month <= 11), 'season'] = 'autumn'
seasonal_time_year.loc[(seasonal_time_year.month >= 12) | (seasonal_time_year.month <= 2), 'season'] = 'winter'


# In[17]:


sns.factorplot(x='year', y='label', hue='season', data=seasonal_time_year, kind='bar')


# **I see no trends in seasonal data**

# In[56]:


train_df.created_utc = pd.to_datetime(train_df.created_utc)
train_df['hour_created'] = train_df.created_utc.dt.hour
train_df.head()


# In[19]:


train_df.month = train_df.month.astype(int)
train_df.year = train_df.year.astype(int)
sns.heatmap(train_df.corr())


# **It can be clearly seen, that the only numerical feature, with which label has correlation is amount of downvotes**

# In[57]:


sns.distplot(train_df.loc[train_df['label'] == 1, 'hour_created'], bins=24)


# **The distribution is quite weird, but we clearly can see the tendency for sarcasm post to appear after 11 am and 
# before 0 am**

# In[21]:


train_df['day_of_week_created'] = train_df.created_utc.dt.dayofweek
sns.distplot(train_df.loc[train_df['label'] == 1, 'day_of_week_created'], bins=7)


# **The trend is monotone, except for the weekends, where the amount of sarcastic comments decreases. Maybe, this happens, because people are usually less exhaused on weekends**

# In[22]:


len(train_df.subreddit.unique()) # there 14876 unique subreddits


# In[23]:


train_df.head()


# Now, let's have a look at some text features

# In[58]:


vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
vectorized_comments = vectorizer.fit_transform(train_df['comment'])
vectorized_comments.shape


# Let's do the same to parent comments

# In[76]:


X_train, y_train = train_df.loc[:,['hour_created']], train_df.loc[:,'label']


# In[77]:


from sklearn.preprocessing import LabelEncoder


# In[79]:


X_train.head()


# In[80]:


from scipy.sparse import hstack


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(hstack([X_train, vectorized_comments]), y_train, random_state=17)


# In[82]:


logit = LogisticRegression(random_state=17, n_jobs=-1, verbose=True, solver='lbfgs')
parameters = {'C' : np.logspace(-2, 2, 5)}


# In[86]:


clf = GridSearchCV(logit, parameters, cv=5)
clf.fit(X_train, y_train)


# In[88]:


logit.fit(X_train, y_train)


# In[91]:


accuracy_score(y_test, clf.predict(X_test))


# In[90]:


accuracy_score(y_test, logit.predict(X_test))


# In[38]:


import eli5
eli5.show_weights(estimator=clf)


# In[75]:


clf.best_params_


# In[ ]:




