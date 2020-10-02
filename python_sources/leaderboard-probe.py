#!/usr/bin/env python
# coding: utf-8

# # Leaderboard Probing
# They mentioned in the coarse that it is possible to probe the leaderboard for some information about the public test set. In this competetion we get to probe both the public and private test sets, but you get a lot more submissions for the public leaderboard. In this notebook we'll take a look at a few basic leaderboard probes for the "Final Project: Predict Future Sales" competetion. 

# # Imports and Load Data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


test = pd.read_csv('../input/test.csv.gz', index_col=False)
sample_submission = pd.read_csv('../input/sample_submission.csv.gz', index_col='ID')
transactions = pd.read_csv('../input/sales_train.csv.gz', index_col=False)


# # Probe for the Mean and Variance
# 
# With the rmse from two constant submissions we can infer the mean and variance of the test data. 

# In[ ]:


all_ones_submission = sample_submission.copy()
all_ones_submission.item_cnt_month = 1
all_ones_submission.head()


# In[ ]:


all_ones_submission.to_csv('../allOnes.csv')


# In[ ]:


all_zeros_submission = sample_submission.copy()
all_zeros_submission.item_cnt_month = 0
all_zeros_submission.head()


# In[ ]:


all_zeros_submission.to_csv('../allZeros.csv')


# ## Calculating the mean
# 
# Let $e_0$ be the rmse from our all zeros probe, $e_1$, be the rmse from our all ones probe, $n$ be the size of the public test set, and let $\mu = \frac{1}{n}\sum{y_i}$ be the mean of the test labels. Then 
# $$e_0{}^2 = \frac{1}{n}\sum{y_i{}^2}$$ and
# 
# $$\begin{eqnarray}
#  e_1{}^2 &=& \frac{1}{n}\sum{(y_i{} - 1)^2} \\
#   &=& \left(\frac{1}{n}\sum{y_i{}^2}\right) - 2\left(\frac{1}{n}\sum{y_i}\right)  + \frac{1}{n}\sum{1} \\
#   &=& \left(\frac{1}{n}\sum{y_i{}^2}\right) - 2\mu  + 1
#  \end{eqnarray}$$
#  
#  Thus
#  
# $$\begin{eqnarray}
# e_1{}^2 - e_0^2 &=& -2\mu + 1 \\
# \frac{e_1{}^2 - e_0^2 - 1}{-2} &=& \mu
# \end{eqnarray} $$

# In[ ]:


# I won't post the results of my submissions as that might constitute unfair 
# teaming but you can duplicate my work.
# It's just as easy to calculate the variance of the test labels, I'll leave
# that to the reader. 


# In[ ]:





# # Probe for the Public Test Set Size
# 
# Seems like something we might like to know.

# In[ ]:


# First of all, we can only probe things that are in the test set

test_shops = test.shop_id.unique()
test_items = test.item_id.unique()

test_transactions = transactions[transactions.item_id.isin(test_items) & transactions.shop_id.isin(test_shops)]


# ## Find an item that probably didn't sell
# If we know that the true label is 0 then we will know how much we've changed the error when we change our prediction for this item. 

# In[ ]:


items_by_month = test_transactions.groupby(['date_block_num', 'item_id']).sum()
items_by_month = items_by_month.unstack('date_block_num')[['item_cnt_day']]
items_by_month.rename(columns={'item_cnt_day':'total_item_cnt'}, inplace=True)
items_by_month.fillna(0, inplace=True)
items_by_month.head()


# In[ ]:


# Let's find some items that didn't sell at all in the last two years of the train
# data. We'll assume they didn't sell at all during the test period as well. 
last_26_months = items_by_month[items_by_month.columns[7:]]
items_by_month[last_26_months.sum(axis=1)==0]


# In[ ]:


# Didn't sell a single one since month 1. This is almost certainly going to be 
# 0 in the test sets. 
probe_item = 13536


# ## Submit the Probe

# In[ ]:


# We're just sort of hoping that this will be in the public test set and not the 
# private one. Run it a few times until the error we get is larger than $e_0$. 
shop_num = np.random.choice(test_shops)
shop_num


# In[ ]:


# We'll fill in a single row with this value
# We want it big for numerical reasons. 
M = 500


# In[ ]:


probe_submission = all_zeros_submission.copy()
probe_submission[(test.item_id==probe_item) & (test.shop_id==shop_num)] = M

probe_submission.to_csv('../testSizeProbe.csv')


# ### Calculate Public Test Set Size
# 
# Let $e_{size}$ be the rmse from our size probe, and let $e_0$, $n$, and $M$ be as above. Then because we have assumed that the true label for our probe item is 0 we know that
# $$e_{size}{}^2 = \frac{1}{n}\left(\left(\sum y_i{}^2\right) + M^2\right)$$
# $$e_{size}{}^2 - e_0{}^2 = \frac{1}{n}M^2$$
# $$n = \frac{M^2}{e_{size}{}^2 - e_0{}^2}$$
# 
# #### Numerical Issues
# Note that we only get 5 decimal digits of precision for $e_0$ and $e_{size}$ so it's bad if they're too close together. For example if $e_0{}^2=1.11111...$ and $e_{size}{}^2 = 1.11113...$ then our best guess is $e_0{}^2 - e_{size}{}^2 = 0.00002...$ and this has only 1 digit of precision! (Python will output more digits, but they're not reliable.) That's not enough to calculate $n$. So we made $M$ larger to make $e_{size}$ more different from $e_0$. 

# In[ ]:


# Again, not posting my results...have fun!


# In[ ]:





# # Probe the Public/Private Test Set Split
# 
# Did the organizers split the test set by shop id, item id, randomly, or something else? Here we'll test whether or not they split by shop id. 
# 
# Assuming we got unlucky at least once above we've discovered a shop/item pair that isn't in the public test set. We'll change the predictions from 0 to 20 for every item in that shop and see if the error changes at all. 

# In[ ]:


# Note that this is just a random shop here - if you know one that's definitely 
# in the private test set you'll want to use that one instead. 
shop_num


# In[ ]:


probe_submission_2 = all_zeros_submission.copy()
probe_submission_2[test.shop_id==shop_num] = 20

probe_submission_2.to_csv('publicPrivateSplitProbe.csv')


# In[ ]:


# That's it! Just check and see if the error here is any different from $e_0$. 


# # Conclusion
# Leaderboard probing isn't real data science, it's a way to game the competition. But we get to do some fun math and if we want to win competitions then we probably need to do it. There's more leaderboard probing that we can do than I put in this notebook, I'm not giving away all my tricks just yet!

# In[ ]:




