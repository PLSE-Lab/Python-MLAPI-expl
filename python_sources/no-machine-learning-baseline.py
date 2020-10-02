#!/usr/bin/env python
# coding: utf-8

# # Simple baseline based on aggregated results for assessments

# In this notebook, we look at train_labels and calculate the average accuracy_group for different assesments in the training data. We then round them to the nearest integer and use them as predictions for the test data.

# ### Average Accuracy Group by Assessment

# In[ ]:


import pandas as pd

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
train_labels.groupby(['title'])['accuracy_group'].mean()


# Clearly some assesments are harder than the others..
# 
# We'll round these values and put them in a dictionary

# In[ ]:


scores_dict = {
    'Bird Measurer (Assessment)': 1,
    'Cart Balancer (Assessment)': 2,
    'Cauldron Filler (Assessment)': 2,
    'Chest Sorter (Assessment)': 1,
    'Mushroom Sorter (Assessment)': 2
}


# ## Get unique users in test data

# In[ ]:


test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
test_users = set(test['installation_id'])


# ## Identify the last assessment for each user and score them

# In[ ]:


get_ipython().run_cell_magic('time', '', "sub = []\nfor u in test_users:\n    user_data = test.loc[test['installation_id']==u, :] \n    user_data = user_data.sort_values(['timestamp']) # sort by time\n    assessment = list(user_data['title'])[-1] # title of the last event (always an assessment)\n    score = scores_dict[assessment] # score the assesment\n    sub.append([u, score])")


# ### Submission

# In[ ]:


sub_df = pd.DataFrame(sub, columns=['installation_id', 'accuracy_group'])
sub_df.to_csv('submission.csv', index=False)

