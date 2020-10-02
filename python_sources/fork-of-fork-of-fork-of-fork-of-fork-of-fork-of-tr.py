#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


my_sub = pd.read_csv('/kaggle/input/mysub-014/submission_small_014.csv')


# In[ ]:


my_sub


# In[ ]:


for i, row in my_sub.iterrows():
    submission.loc[submission['installation_id'] == row['installation_id'], 'accuracy_group'] = row['accuracy_group']

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission


# In[ ]:


#my_sub.shape


# In[ ]:


#sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


# sample_submission['accuracy_group'] = preds.astype(int)
# sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


# sample_submission['accuracy_group'].value_counts(normalize=True)


# In[ ]:




