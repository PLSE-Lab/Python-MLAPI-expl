#!/usr/bin/env python
# coding: utf-8

# ## What this notebook does:
# - Map `installation_id` in `sample_submission.csv` to the correspoding assessment.
# - Compute the average accuracy of each assessment using `train_labels.csv`.
# - Estimate `accuracy_group` from the average accuracy.

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().run_line_magic('ls', '-lh ../input/data-science-bowl-2019/')


# In[ ]:


test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
test.head()


# In[ ]:


last_event = test.sort_values(['installation_id', 'timestamp']).groupby('installation_id').last().reset_index()
last_event.head()


# In[ ]:


ends_with_assessment = last_event['title'].str.contains('Assessment')
last_event[~ends_with_assessment]


# In[ ]:


sbm_sample = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
sbm_sample = pd.merge(sbm_sample, last_event[['installation_id', 'title']], on='installation_id')
sbm_sample


# In[ ]:


labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
labels.head()


# In[ ]:


def predict(accuracy):
    if accuracy > 0.5:
        return 3
    
    if accuracy > 0.4:
        return 2
    
    if accuracy > 0.13:
        return 1
    
    return 0


# In[ ]:


agg = labels.groupby('title').sum()[['num_correct', 'num_incorrect']].reset_index()
agg['accuracy'] = agg['num_correct'] / (agg['num_incorrect'] + agg['num_correct'])
agg['accuracy_group'] = agg['accuracy'].map(predict)
agg


# In[ ]:


sbm = pd.merge(sbm_sample.drop('accuracy_group', axis=1), agg, on='title')
sbm


# In[ ]:


sbm[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)


# In[ ]:




