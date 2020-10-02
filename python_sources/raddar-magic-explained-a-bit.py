#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Raddar astonished us with his insights on purchase amount and other features in his seminal kernel: https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights .  I decided to have a further look to see if one could have found it without resorting to optimization procedures and manual tweaking.  Here is what I got.  Hope it will please some. 
# 
# Disclaimer: this is only providing some methodology tips. It does not reuslt in insights that Raddar did not disclose.  I prefer to be clear on that given we are in the last week period.

# We will focus on purchase amount. Let's load all data we have.

# In[ ]:


historical_transactions = pd.read_csv('../input/historical_transactions.csv', usecols=['purchase_amount'])
new_transactions = pd.read_csv('../input/new_merchant_transactions.csv', usecols=['purchase_amount'])
data = pd.concat((historical_transactions, new_transactions))


# This data has been normalized as Raddar claimed.  Indeed, its mean is close to 0:

# In[ ]:


data.purchase_amount.mean()


# Purchase amounts should be non negative.  My preferred Brazil expert, namely my team mate Giba, told me there is no minimum on purchase amount on cards in Brazil, hence we can start with a minimum at 0:

# In[ ]:


data['new_amount'] = (data.purchase_amount - data.purchase_amount.min())


# Let's look at the different amounts we have in the data.

# In[ ]:


s = data.groupby('new_amount').new_amount.first().to_frame().reset_index(drop=True)
s.head(10)


# Let's now compute the successive differences between these amounts.  These should be multiples of cents if purchase real unit is a monetary unit like the Real.

# In[ ]:


s['delta'] = s.new_amount.diff(1)
s.head(10)


# We see that the deltas seem constant.  For how long are they constant?  One way to find out is to look at when delta is higher:

# In[ ]:


s[s.delta > 2e-5].head()


# Allright, deltas are small until index 5263.  Let's focus on that part.  We also get rid of the first row as its delta was undefined.

# In[ ]:


s = s[1:52623]


# In[ ]:


s.head()


# In[ ]:


s.tail()


# In[ ]:


s.delta.mean()


# This is the smallest posisble difference between two amounts in our data.  If this is one cent, then we can retrieve the orignal purchase amounts by a simple division:

# In[ ]:


data['new_amount'] = data.new_amount / (100 * s.delta.mean())


# Let's look at most frequent values.

# In[ ]:


data.new_amount.value_counts().head(10)


# Pretty close to integers, aren't they?

# If these numbers represent money, then they should be rounded to 2 decimals.

# In[ ]:


data['two_decimal_amount'] = np.round(data.new_amount, 2)


# How much did we round?

# In[ ]:


np.abs(data.two_decimal_amount - data.new_amount).mean()


# Not much... :)

# How many integers do we get?

# In[ ]:


(data.two_decimal_amount == np.round(data.two_decimal_amount)).mean()


# Nearly 40%.

# Is this better than Raddar's kernel?  He computed this for new transactions only.  Let's see what we get.

# In[ ]:


tmp = data[-new_transactions.shape[0]:]
(tmp.two_decimal_amount == np.round(tmp.two_decimal_amount)).sum()


# This is exactly the same number that Raddar had in his kernel.

# The only added value of mine, if any, is that we derived these in a quite straightforward way.  Hope you enjoyed it.

# In[ ]:




