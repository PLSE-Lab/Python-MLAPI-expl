#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


leaks = pd.read_csv('/kaggle/input/leakaggregator/leaked_test_target.csv')
submission = pd.read_csv('/kaggle/input/bland-by-leak/submission_blend.csv')


# In[ ]:


(~leaks['meter_reading'].isna()).sum()


# In[ ]:


leaks = leaks.sort_values('row_id')
submission = submission.sort_values('row_id')

leaks = leaks.set_index(leaks['row_id'])
submission = submission.set_index(submission['row_id'])


# In[ ]:


submission.loc[~leaks['meter_reading'].isna(), 'meter_reading'] = leaks.loc[~leaks['meter_reading'].isna(), 'meter_reading']


# In[ ]:


submission.to_csv('bland_with_leaked.csv',index=False)

