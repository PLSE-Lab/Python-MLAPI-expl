#!/usr/bin/env python
# coding: utf-8

# **Let's investigate "D" features!**
# 
# We all know that the data provided in this competition consist of transactions. I was curious about how to correctly identify the same cardholder/card and group that transactions. Organizers revealed that "D" features contain information about different time deltas and "card" features apply to a payment card info. 
# 

# In[ ]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')


# Let's group by 'card' features and look closely on those combinations that give us around 5-10 rows.

# In[ ]:


by = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
grouped = train.groupby(by, as_index=False)['TransactionID'].count()
grouped[grouped['TransactionID']==7].head(5)


# In[ ]:


# This combination of cardx features gives 7 rows.
card1 = 18383
card2 = 128
card3 = 150
card4 = 'visa'
card5 = 226
card6 = 'credit'

train_slice = train[(train['card1']==card1)&
                   (train['card2']==card2)&
                   (train['card3']==card3)&
                   (train['card4']==card4)&
                   (train['card5']==card5)&
                   (train['card6']==card6)]


# Now let's look closer at these rows.

# In[ ]:


features = ['TransactionID','TransactionDT','ProductCD', 'P_emaildomain', 'R_emaildomain'
            , 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
train_slice = train_slice.sort_values(['TransactionID'])[features]
train_slice


# Now we can add "DaysFromStart" column by divining TransactionDT on 60*60*24 and then round it to get a number of days from a starting point.

# In[ ]:


train_slice['DaysFromStart'] = np.round(train_slice['TransactionDT']/(60*60*24),0)


# In[ ]:


train_slice


# **Time to identify what's behind 'D' columns!**
# 
# Create feature:
#     DaysFromPreviousTransaction = DaysFromStart[row_(i)] - DaysFromStart[row_(i-1)]

# In[ ]:


train_slice['DaysFromPreviousTransaction'] = train_slice['DaysFromStart'].diff()


# In[ ]:


features = ['TransactionID', 'TransactionDT', 'D1', 'D2', 'D3', 'DaysFromStart', 'DaysFromPreviousTransaction']
train_slice[features].iloc[3:]


# I can be wrong but I believe these transactions belong to the same user. One can see that DaysFromPreviousTransaction is equal to D3 which drives me to think that **D3 indicates number of days from the previous transaction**.
# 
# Also D1 is cumulatively increasing and for example 481 = 449 + 32 and 510 = 481 + 29, i.e. **D1 could indicate days from the first transaction**. 
# 
# D2 is almost always equal to D1 but for the first transaction when D1 is equal to 0 D2 is nan.

# **If I'm not wrong this should be a useful knowledge to identify users and proceed with a meaningful FE.**
# 
# ** Please share your thoughts! **
