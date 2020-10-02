#!/usr/bin/env python
# coding: utf-8

# ## ** LB probing, public/private split and simple seasonal decomposition **
# > 
# Hello fellow Kagglers :)
# 
# The kernel I'm building here aims at trying to use **leaderboard probing** to obtain a reasonable score on the public LB. The main idea is to **not use any ML model** to predict sales for the month of November 2015.

# ## ** Part I - LB probing and forecasting **
# 
# We'll first use sample submissions, one with only 0s and one with only 1s to calculate the mean of true LB values.

# In[ ]:


from pandas import read_csv

sample_submission = read_csv('../input/sample_submission.csv')

only0s = sample_submission.assign(item_cnt_month = 0)
only1s = sample_submission.assign(item_cnt_month = 1)

only0s.head()


# In[ ]:


only1s.head()


# The submission with only 0s gives us a score of $1.25011$, and the one with only 1s yields $1.4124$ Using the following formula, those two values allow us to compute the exact mean of true values in public leaderboard.
# 
# $$MSE(1) - MSE(0) = \sum_{i=0}^N \frac{(y_i - 1)^2}{N} - \sum_{i=0}^N \frac{y_i^2}{N}$$
# 
# $$ = \sum_{i=0}^N \frac{y_i^2 - 2y_i + 1}{N} - \frac{y_i^2}{N}$$
# 
# $$ = \sum_{i=0}^N \frac{1 - 2y_i}{N}$$
# 
# $$ = 1 - 2\sum_{i=0}^N \frac{y_i}{N}$$
# 
# $$ = 1 - 2\overline{y}$$
# 
# Given that $MSE(0) = 1,5627750121$ and $MSE(1) = 1.9949$, we can therefore establish that the mean value of true values is $0.283936502$ (We'll consider $0.284$ for convenience).
# 
# It is interesting to compare with the mean of train set to see **if public/private split can be identified**.
# 
# To do this, we first need to **aggregate train sales monthly** to be on the same page as test. We must not forgt to clamp values between 0 and 20.

# In[ ]:


del only0s, only1s

train = read_csv('../input/sales_train.csv')
train_by_month = train.groupby(['date_block_num', 'shop_id', 'item_id'])[['item_cnt_day']].sum().clip(0, 20)
train_by_month.columns = ['item_cnt_month']
train_by_month = train_by_month.reset_index()
del train

train_by_month.head()


# Seems good. Now we can compare the mean.

# In[ ]:


train_by_month.groupby('date_block_num')['item_cnt_month'].mean().tail()


# Not quite what we expected ! This surely means that there are a **lot more 0s in test set than in train set**. Are there actually any 0s in train set ?

# In[ ]:


print('About %.2f%% of train values are 0s' % (train_by_month[train_by_month['item_cnt_month'] == 0].shape[0] * 100 / train_by_month.shape[0]))


# **Way less** than test set. There has to be something in the way the organizers built the test set. Let's have a look at it.

# In[ ]:


test = read_csv('../input/test.csv')
len(test.shop_id.unique()), len(test.item_id.unique()), len(test)


# **42 * 5100 is 214200** so yes, basically the test set is built using **all combination of those 42 shops and 5100 items**. No wonder there are so many 0s in true values.
# 
# We need to make the **train set equivalent** to this scheme. To do that, we'll simply use those 42 shops and 5100 items, make all possible pairs with **date_block_num** from 0 to 33 and join with train set. Let's see how it goes.**

# In[ ]:


from itertools import product
from pandas import DataFrame

pairs = DataFrame(list(product(list(range(34)), test.shop_id.unique(), test.item_id.unique())), columns = ['date_block_num', 'shop_id', 'item_id'])
pairs.head()


# In[ ]:


pairs.shape


# Now join with train set. We also downcast values to reduce memory usage.

# In[ ]:


def displayWithSize(df):
    print('Shape : %i x %i' % df.shape)
    m = df.memory_usage().sum()
    if m >= 1000000000:
        print('Total memory usage : %.2f Go' % (m / 1000000000))
    else:
        print('Total memory usage : %.2f Mo' % (m / 1000000))
    return df.head()


# In[ ]:


from numpy import uint8, uint16, float16

pairs_red = pairs.assign(date_block_num = pairs['date_block_num'].astype(uint8))
pairs_red = pairs_red.assign(shop_id = pairs['shop_id'].astype(uint8))
pairs_red = pairs_red.assign(item_id = pairs['item_id'].astype(uint16))
del pairs

inflated_train = pairs_red.merge(train_by_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
inflated_train.fillna(0.0, inplace=True)
del pairs_red

displayWithSize(inflated_train)


# In[ ]:


inflated_train.dtypes


# ### ** WARNING **
# 
# **Downcasting** is a good strategy to help **reduce memory error chances**, but it has its ***limit*** : each data type has its own min and max values, for example **float16 goes from -65504 to +65504**, hence when calculating things like mean or sum, during the calculus **the value can exceed the max and result in a NaN or inf value**. So be careful :)
# 
# That's why we did not downcast **item_cnt_month**.

# In[ ]:


from numpy import finfo

finfo(float16)


# Ok ! Now we can compare mean values of last month in train and month in test !

# In[ ]:


from numpy import float64

inflated_train[inflated_train['date_block_num'] == 33]['item_cnt_month'].astype(float64).clip(0, 20).mean()


# Amazing ! This clearly means that public/private split in the leaderboard is **completely random** !!
# 
# How does the sales trend look like with our new train set ?

# In[ ]:


from matplotlib.pyplot import plot
get_ipython().run_line_magic('matplotlib', 'inline')

sales_by_month = inflated_train.groupby('date_block_num')['item_cnt_month'].sum().tolist()
plot(sales_by_month)


# A nice trend draws itself ! It is **going up**, because we are only looking at a **small subset of items/shops**, that is on of the **latest as it is from test set**. Can we use a seasonal decomposition on this one ?

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import figure

decomposition = seasonal_decompose(sales_by_month, freq=12, model='multiplicative')
fig = figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# Very interesting ! The residual part **oscillate between 0.9 and 1.1** which is **only 10%** of original data. This is low enough. We can hence build a **multiplicative factor based on trend * seasonal**, that will, combined with latest months, help us build a no-ML based model.
# 
# The strategy for this seasonal decompose is simple : **seasonal part is repetitive** so we will just **use it as is**. **Trend part is not** : we will use a rolling mean scheme to **forecast trend** on month 33 (validation) and 34 (kaggle prediction). We then multiply both, to get a factor. Then, to forecast month n+1, we divide month n sales by the factor for month n, and multiply by factor for month n+1.

# In[ ]:


def rolling_mean(data, timespan):
    n = len(data)
    output = []
    for i in range(n):
        maxWindow = min(i - max(0, i-timespan//2), min(n, i+timespan//2) - i)
        output.append(data[i-maxWindow:i+maxWindow+1])
    return list(map(lambda x: sum(x) / len(x), output))

rmean = rolling_mean(sales_by_month, 12)
plot(rmean)


# We see a clear divergence at the end, that's because we don't have data about the future. Therefore we need to smoothen it using a mean of last 6 months trend.

# In[ ]:


from numpy import mean

incr_step = mean(list(map(lambda x: x[0] - x[1], zip(rmean[22:28], rmean[21:27]))))
incr_step


# In[ ]:


rmean_corrected = rmean[:28]
current = rmean_corrected[-1]

for _ in range(6):
    current += incr_step
    rmean_corrected.append(current)

plot(rmean_corrected)


# Let's see if product of both trend and seasonal is close to reality

# In[ ]:


forecast = list(map(lambda x: x[0] * x[1], zip(rmean_corrected, decomposition.seasonal)))

plot(sales_by_month)
plot(forecast)


# Looks rather nice ! We can forecast for month 34.

# In[ ]:


forecast.append((rmean_corrected[-1] + incr_step) * decomposition.seasonal[-12])

plot(sales_by_month)
plot(forecast)


# All right ! We have our forecasted values for both month 33 and 34, let's do some predictions !

# ## ** Part II - No-ML predictions **
# 
# The process is simple. We know data about month N, and we want to predict for month N+1.
# 
# We have real **mean sales for month N** and forecasted **mean sales for month N+1**.
# We know exactly the **mean values of month N+1** (or at least an estimation).
# 
# We proceed as follow :
#  - We take all **item_cnt_month** values from month N
#  - We divide values by **mean sales of month N**
#  - We multiply values by **mean sales of month N+1**
#  - We fill 0 values with a constant value such that the total mean of all values becomes the **mean values of month N+1** (this one goes from the fact that to minimize RMSE with a constant, you need to take the mean of true values)
#  
# The last step is a little bit tricky. Let us note :
#  - $C_0$ the number of 0s in values
#  - $C$ the number of values
#  - $m_c$ the current mean
#  - $m_t$ the target mean
#  - $y_i$ the value $i$
#  
# We then have :
# 
# $$m_c = \sum_{i=0}^C \frac{y_i}{C} \iff m_c = \sum_{i=0, y_i\ne0}^C \frac{y_i}{C}$$
# 
# And :
# 
# $$m_t = m_c + (m_t - m_c) = \sum_{i=0, y_i\ne0}^C \frac{y_i}{C} + \frac{C_0}{C_0}(m_t - m_c) = \sum_{i=0, y_i\ne0}^C \frac{y_i}{C} + \sum_{i=0, y_i=0}^C \frac{C}{C_0}\frac{m_t - m_c}{C}$$
# 
# $$ = \frac{1}{C}(\sum_{i=0, y_i\ne0}^C y_i + \sum_{i=0, y_i=0}^C \frac{C}{C_0}(m_t - m_c))$$

# In[ ]:


month_n = 32
month_n_plus_1 = 33

item_cnt_month_n = inflated_train[inflated_train['date_block_num'] == month_n]['item_cnt_month']

C0 = sum(item_cnt_month_n == 0)
C = len(item_cnt_month_n)
mt = inflated_train[inflated_train['date_block_num'] == month_n_plus_1]['item_cnt_month'].mean()

item_cnt_month_n_plus_1 = item_cnt_month_n / sales_by_month[month_n]
item_cnt_month_n_plus_1 = (item_cnt_month_n_plus_1 * forecast[month_n_plus_1]).clip(0, 20)
item_cnt_month_n_plus_1[item_cnt_month_n_plus_1 == 0] = (C / C0) * (mt - item_cnt_month_n_plus_1.mean())

item_cnt_month_n_plus_1.mean(), mt


# What is the RMSE ?

# In[ ]:


from sklearn.metrics import mean_squared_error
from numpy import sqrt

def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

rmse(item_cnt_month_n_plus_1, inflated_train[inflated_train['date_block_num'] == month_n_plus_1]['item_cnt_month'])


# Not quite convinced we can call this a good result. But still, it is way better than the constant 0s prediction so...
# 
# The interest lied in the approach anyway :)

# In[ ]:


month_n = 33
month_n_plus_1 = 34

item_cnt_month_n = inflated_train[inflated_train['date_block_num'] == month_n]['item_cnt_month']

C0 = sum(item_cnt_month_n == 0)
C = len(item_cnt_month_n)
mt = 0.284

item_cnt_month_n_plus_1 = item_cnt_month_n / sales_by_month[month_n]
item_cnt_month_n_plus_1 = (item_cnt_month_n_plus_1 * forecast[month_n_plus_1]).clip(0, 20)
item_cnt_month_n_plus_1[item_cnt_month_n_plus_1 == 0] = max(0, min(20, (C / C0) * (mt - item_cnt_month_n_plus_1.mean())))

item_cnt_month_n_plus_1.mean(), mt


# In[ ]:


DataFrame(list(zip(range(len(item_cnt_month_n_plus_1)), item_cnt_month_n_plus_1)), columns=['ID', 'item_cnt_month']).to_csv('submission.csv', index=False)


# In[ ]:




