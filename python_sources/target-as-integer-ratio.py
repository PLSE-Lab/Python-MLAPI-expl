#!/usr/bin/env python
# coding: utf-8

# # Converting target values to fractions
# 
# Following the insights from this awesome kernel 
# [https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights](http://) 
# 
# I wrote this small kernel which will automatically try to convert all target values from train data into rational fractions with a given precision.
# 

# In[ ]:


import numpy as np
import pandas as pd
from fractions import Fraction
import gc


# In[ ]:


train = pd.read_csv('../input/train.csv')


# As was explained in original above kernel, all target values in train data set were actually transformed using log2 function. 
# 
# The original values can be obtained by applying 2^target backtransformation

# In[ ]:


train.drop(['first_active_month','feature_1','feature_2','feature_3'],axis=1, inplace=True)
train.set_index(['card_id'],inplace=True)
train['target_original'] = 2**train['target']
train.head()


# As it turnes out some of the original target values in target_original column are really close to ratios like 2/5, 8/9 etc.
# 
# The function below will calculate the smallest fraction which comes close enough to original target value whith a given numerical precision

# In[ ]:


def calcRatio(df, column, precision):
    limit = 10
    
    frac = 'fraction'
    dfrac = 'decimal'
    diff = 'diff'
    
    df[frac] = df[column].apply(lambda x: Fraction(x).limit_denominator(limit))
    df[dfrac] = df[frac]*1.
    df[diff] = abs(df[column]-df[dfrac])
    mask = df[diff] > precision
    
    while len(df[mask]) > 0:
        limit = limit*10
        df.loc[mask, frac] = df[mask][column].apply(lambda x: Fraction(x).limit_denominator(limit))
        df[diff] = abs(df[column]-df[frac]*1.)
        mask = df[diff] > precision 

    df[dfrac] = df[frac]*1.
    df['numerator']=df[frac].apply(lambda x: x.numerator)
    df['denominator']=df[frac].apply(lambda x: x.denominator)


# Now lets figure out fraction values which were used to calculate original target values.

# In[ ]:


calcRatio(train, 'target_original', 10**(-6))


# As you can see below lots of original target values at the lower end can be matched pretty well by fractions with small denominator. I exclude the outliers where ratio = 0 and list some values just to get an impression how many target values can be matched pretty well by fractions with low denominator.
# 
# Column 'decimal' contains decimal value of determined fraction and 'diff' the difference to original target value

# In[ ]:


train.sort_values(['numerator','denominator'], inplace=True)
train[train.fraction!=0][2400:2450]


# In[ ]:


train.sort_values(['denominator'], inplace=True)
train[train.fraction!=0][2400:2450]


# In[ ]:


calcRatio(train, 'target_original', 10**(-3))


# In[ ]:


train.sort_values(['card_id'], inplace=True)
train[(train.fraction!=0)&(train['diff']<(10**(-7)))]


# I hope this may lead to more insights and can be used for postprocessing.

# There was a hypothesis that some of the denominator/numerator values may match a number of authorized transactions from new/historical data. Lets check it. I'll import these data and join with above fractionized target set

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


usecols = ['authorized_flag','card_id','installments','merchant_id']


# In[ ]:


hist_xfer = pd.read_csv('../input/historical_transactions.csv', usecols=usecols )
hist_xfer['new']=0
hist_xfer = reduce_mem_usage(hist_xfer, True)
new_xfer = pd.read_csv('../input/new_merchant_transactions.csv', usecols=usecols)
new_xfer['new']=1
new_xfer = reduce_mem_usage(new_xfer, True)


# In[ ]:


xfer = pd.concat([new_xfer,hist_xfer])
del new_xfer, hist_xfer
gc.collect()


# In[ ]:


xfer.reset_index(drop=True)
xfer['authorized_flag'] = xfer['authorized_flag'].map({'Y': 1, 'N': 0}).astype(np.int16)


# In[ ]:


xfer['auth_new']=xfer['authorized_flag']*xfer['new']
xfer['auth_hist']=abs(xfer['authorized_flag']*(xfer['new']-1))
xfer.head()


# In[ ]:


aggs= {'installments': ['sum'],
       'authorized_flag': ['sum'],
       'merchant_id': ['nunique', 'count'],
       'auth_new': ['sum'],
       'auth_hist': ['sum']
}


# In[ ]:


df = xfer.groupby('card_id').agg(aggs)
df.columns = df.columns.map('_'.join)
df['max'] = df[['installments_sum', 'merchant_id_count']].max(axis=1)
df.head()


# In[ ]:


train = train.join(df, on='card_id')


# In[ ]:


calcRatio(train, 'target_original', 10**(-6))
train.sort_values(['denominator'], inplace=True)


# Lets check if there are some matches for denominator and authorized new transactions

# In[ ]:


display_col=['target','target_original','fraction','decimal','numerator','denominator','auth_new_sum','auth_hist_sum']
mask = (train['denominator'] == train['auth_new_sum']) & (train['diff']<10**(-4)) & (train['fraction']!=0)
train[mask][display_col].head(10)


# In[ ]:


train[mask][display_col].tail(10)


# In[ ]:


print(len(train[mask]))


# There are some matches, but mostly for denominator == 1 similar situation for numerator

# In[ ]:


mask = (train['numerator'] == train['auth_new_sum'])  &  (train['diff']<10**(-4)) &(train['fraction']!=0)
train[mask][display_col].head(10)


# In[ ]:


train[mask][display_col].tail(10)


# In[ ]:


print(len(train[mask]))


# But even if there would have been more matches: a fraction 1/3 could have been calculated by 10/33, 20/60. Looks like these matches of numerator and number of auth transactions happened just by coincidence
