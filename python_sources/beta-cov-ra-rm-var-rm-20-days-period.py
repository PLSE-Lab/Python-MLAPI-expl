#!/usr/bin/env python
# coding: utf-8

# This kernel show some dependencies for beta risk factor and 20-days covariance of returnsOpenPrevRaw10

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()


# ** Let's find the smallest returnsOpenPrevRaw1**

# In[ ]:


market_train_df.returnsOpenPrevRaw1.min()


# In[ ]:


market_train_df[
    market_train_df.returnsOpenPrevRaw1 == market_train_df.returnsOpenPrevRaw1.min()
].T


# In[ ]:


features = ['time', 'open', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']


# In[ ]:


market_train_df[
    (market_train_df.assetCode == 'PBRa.N') & 
    (market_train_df.time >= '2007-05-01') &
    (market_train_df.time <= '2007-06-20')
][features]


# As we see, **in 20 days period from 2007-05-18 to 2007-06-15** market residual return is much volative than raw return.
# 
# So, beta may depends on std, variance or covariance. One of the common formula is:
# 
# <pre>Beta = Cov (Ra, Rm) / Var(Rm)</pre>
# 
# Let's calculate it for 20 days

# First, we may approximate market rerurn as median of all returns

# In[ ]:


market_train_df['raw_median'] = market_train_df.groupby('time').returnsOpenPrevRaw10.transform('median')


#  <pre>Cov(x,y) = SUM [(xi - x_mean) * (yi - y_mean)] / (n - 1) = 
#  = (SUM(xi * yi) - SUM(xi * y_mean) - SUM(yi * x_mean) + SUM (x_mean * y_mean)) / (n-1) =
#  = (MEAN(xi * yi) - x_mean * y_mean - y_mean * x_mean + x_mean * y_mean) * n / (n-1) =
#  = (MEAN(X * Y) - MEAN(X) * MEAN(Y)) * n / (n-1)
#  </pre>

# In[ ]:


market_train_df['xy'] = market_train_df.returnsOpenPrevRaw10 * market_train_df.raw_median

roll = market_train_df.groupby('assetCode').rolling(window=20)

market_train_df['cov_xy'] = (
    (roll.xy.mean() - roll.returnsOpenPrevRaw10.mean() * roll.raw_median.mean()) * 20 / 19
).reset_index(0,drop=True)


# Then we calculate the variance of the market:

# In[ ]:


market_train_df['var_y'] = roll.raw_median.var().reset_index(0,drop=True)


# And calculate beta with shift(1)

# In[ ]:


market_train_df['beta'] = (market_train_df['cov_xy'] / market_train_df['var_y'])
market_train_df['beta'] = market_train_df.groupby('assetCode')['beta'].shift(1)


# Let's check results

# In[ ]:


market_train_df[
    (market_train_df.assetCode == 'PBRa.N') & 
    (market_train_df.time >= '2007-05-01') &
    (market_train_df.time <= '2007-06-20')
][features+['beta']]


# As we see, we have large beta values in the periods where we have large market residual returns. Surely, these formula can not be totally right since we had approximations but it looks reasonable.

# In[ ]:




