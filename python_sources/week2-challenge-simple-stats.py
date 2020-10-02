#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train['month'] = pd.DatetimeIndex(train['Date']).month
test['month'] = pd.DatetimeIndex(test['Date']).month


# In[ ]:


mean_vals = train.groupby(['Store', 'Dept', 'month', 'IsHoliday']).median()


# In[ ]:


mean_vals.dtypes


# In[ ]:


merged = test.merge(mean_vals,
                  how = 'left',
                  left_on = ['Store', 'Dept', 'month', 'IsHoliday'],
                  right_index = True,
                  sort = False,
                  copy = False)


# In[ ]:


index = pd.DataFrame({'id':merged['Store'].map(str) + '_' + merged['Dept'].map(str) + '_' + merged['Date'].map(str)})


# In[ ]:


submission = merged.join(index)


# the submission dataframe includes also columns not needed for the challenge. deleted them in excel to be faster. also, there are some empty values that block the upload of the challenge. substituted NaN with a find & replace

# In[ ]:


submission.to_csv('submission.csv')

