#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


test = pd.read_csv('../input/test.csv', nrows=10)


# In[ ]:


last = test.Sequence.apply(lambda x: pd.Series(x.split(','))).mode(axis=1).fillna(0)


# In[ ]:


submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})
submission.head(10)
#submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})


# In[ ]:




