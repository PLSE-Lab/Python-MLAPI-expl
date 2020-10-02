#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train  = pd.read_csv('/kaggle/input/blue-book-for-bulldozer/Train/Train.csv', low_memory=False,
                       parse_dates=['saledate'])


# In[ ]:


df_train.shape


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_train.tail().T)


# In[ ]:


display_all(df_train.describe(include='all').T)


# In[ ]:


df_train.SalePrice = np.log(df_train.SalePrice)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df_train.drop(['SalePrice'], axis=1), df_train.SalePrice)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.issubdtype')


# In[ ]:


def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, 
                                     infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 
            'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 
            'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(df_train, 'saledate')


# In[ ]:


display_all(df_train.head())


# In[ ]:


get_ipython().run_line_magic('pinfo', 'df_train.items')


# In[ ]:


df_train['fiModelDesc']


# In[ ]:


from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
def train_cats(df):
    for n,c in df.items():
            if is_string_dtype(c): 
                df[n] = c.astype('category').cat.as_ordered()


# In[ ]:


train_cats(df_train)


# In[ ]:


#display_all(df_train.dtypes)


# In[ ]:


df_train.UsageBand.cat.set_categories(['High', 'Medium', 'Low'],
    ordered=True, inplace=True)


# In[ ]:


display_all(df_train.isnull().sum().sort_values(ascending=False)/len(df_train))


# In[ ]:


df_train.to_csv('training_data.csv',index=False)


# In[ ]:


df_train.ProductGroup.cat.codes

