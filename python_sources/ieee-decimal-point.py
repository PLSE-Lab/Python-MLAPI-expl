#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math


# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',
                                usecols=['TransactionID', 'TransactionAmt'])


# In[ ]:


train_transaction.head()


# In[ ]:


train_transaction['TransactionAmtDecimalPoint'] = [math.modf(v)[0] for v in train_transaction['TransactionAmt']]


# In[ ]:


train_transaction.head()


# In[ ]:




