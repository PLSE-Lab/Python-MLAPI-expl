#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


restaurants_df = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')
reviews_df = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')


# In[ ]:


restaurants_df = restaurants_df.rename(columns={'Name': 'Restaurant'})
restaurants_df.head()


# In[ ]:


reviews_df['Rating'].unique()


# In[ ]:


reviews_df.head()


# In[ ]:


restaurants_df.shape


# In[ ]:


reviews_df.shape


# One restaurant is associated with multiple reviews.

# In[ ]:


rev_res_df = pd.merge(restaurants_df, reviews_df, on=['Restaurant'], how='inner')
rev_res_df


# # Prepare

# In[ ]:


rev_res_df[rev_res_df['Rating'].isna()]


# In[ ]:


rev_res_df = rev_res_df[~rev_res_df['Rating'].isna()]
rev_res_df.head()


# In[ ]:


rev_res_df['Cost'] = rev_res_df['Cost'].apply(lambda c: c.replace(',',''))
rev_res_df['Cost'] = rev_res_df['Cost'].apply(lambda c: int(c))


# In[ ]:


from sklearn import preprocessing


rev_res_df.loc[rev_res_df['Rating']=='Like', 'Rating'] = '5.5'
rev_res_df['Rating'] = rev_res_df['Rating'].apply(lambda r: float(r))
rev_res_df['Rating'] = preprocessing.minmax_scale(rev_res_df['Rating'], feature_range=(1, 10))
rev_res_df['Rating'] = rev_res_df['Rating'].apply(lambda r: int(r))


# # EDA

# In[ ]:


sns.countplot(x="Rating", data=rev_res_df)


# In[ ]:


sns.regplot(y="Cost", x="Rating", data=rev_res_df)


# In[ ]:


sns.regplot(y="Cost", x="Pictures", data=rev_res_df)


# In[ ]:


sns.regplot(y="Rating", x="Pictures", data=rev_res_df)


# In[ ]:




