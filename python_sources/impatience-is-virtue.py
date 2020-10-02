#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/yelp-reviews-csv"))

# Any results you write to the current directory are saved as output.


# In[2]:


review_set = pd.read_csv("../input/yelp-reviews-csv/yelp_review.csv")


# In[3]:


review_set.set_index(pd.to_datetime(review_set['date']),inplace=True)


# In[4]:


review_set.drop(['review_id','user_id','business_id','date'],axis=1,inplace=True)


# In[5]:


review_set.sort_index(inplace=True)


# In[6]:


daily_mean=review_set.groupby(review_set.index)['stars'].rolling(365).mean()


# In[7]:


review_set['stars'].rolling(20).mean().plot(figsize=(20,10))


# In[ ]:





# In[30]:


slow_df=pd.DataFrame(columns=['stars','text','useful','funny','cool'])


# In[31]:


slow_df=review_set[review_set['text'].str.contains("slow")]
fast_df=review_set[review_set['text'].str.contains("fast")]


# In[32]:


slow_daily_mean = slow_df.groupby(slow_df.index)['stars'].rolling(20).mean()
fast_daily_mean = fast_df.groupby(fast_df.index)['stars'].rolling(20).mean()


# In[33]:


review_set['stars'].rolling(20).mean().plot(alpha=0.9)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[34]:


slow_df['stars'].rolling(20).mean().plot()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[35]:


fast_df['stars'].rolling(20).mean().plot()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[36]:


slow_df['stars'].rolling(20).mean().plot(alpha=0.9,figsize=(20,10))
review_set['stars'].rolling(20).mean().plot(alpha=0.3)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[52]:


fast_df['stars'].rolling(20).mean().plot(alpha=0.9,figsize=(20,10))
review_set['stars'].rolling(20).mean().plot(alpha=0.3)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[37]:


review_set['stars'].describe()


# In[38]:


slow_df['stars'].describe()


# In[39]:


from scipy import stats


# In[40]:


sns.distplot(slow_df['stars'].rolling(20).mean().dropna())
sns.distplot(review_set['stars'].rolling(20).mean().dropna())
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[41]:


import statsmodels.api as sm
df = slow_df.reset_index()


# In[ ]:





# In[42]:


type(df['date'])


# In[43]:


df.head()


# 

# In[44]:


import matplotlib.dates as mdates


# In[45]:


import statsmodels.api as sm


# In[46]:


slow_df['Date'] = slow_df.index.map(mdates.date2num)
slow_df.head()


# In[47]:


slow_df['rolling']=slow_df['stars'].rolling(20).mean()


# In[48]:


slow_df.dropna(inplace=True)


# In[49]:


sns.regplot(x='Date', y='rolling', data=slow_df)


# In[ ]:





# In[ ]:





# In[50]:


model=sm.OLS(endog=slow_df['Date'],exog=slow_df['rolling'])
results = model.fit()


# In[51]:


results.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




