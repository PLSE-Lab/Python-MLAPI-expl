#!/usr/bin/env python
# coding: utf-8

# **Keeping the tradition, here it is**.
# 

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

del train['y']
del train['ID']
del test['ID']

test1 = test.drop_duplicates(keep='first')
print('removed from test:', len(test) - len(test1))

train1 = train.drop_duplicates(keep='first')
print('removed from train:', len(train) - len(train1))

full = pd.concat([train1, test1])
full.drop_duplicates(keep='first', inplace=True)
dups = len(train1) + len(test1) - len(full)
print('test/train dups:', dups)

print(dups/len(test))


# Another interesting fact observing xgboost models built in other kernels is that they're basically 5 features:
# X279, X314, X315, X119, X47.

# In[ ]:




