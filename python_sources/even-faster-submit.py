#!/usr/bin/env python
# coding: utf-8

# # Even Faster Submit
# Based on the discussion by @Peter: https://www.kaggle.com/c/data-science-bowl-2019/discussion/120840.
# 
# *"For quick experimenting, you can train your models locally, predict the available public test set, and ignore the private part."
# *
# 
# This kernel takes it a little further by allowing you to simply run a function locally to get a string to copy/paste into the notebook and test.

# In[ ]:


import pandas as pd

def get_solution_string(sub: pd.DataFrame) -> str:
    """
    Run this in your notebook with your test submission as input
    to get the output string and copy/paste into the cell below.
    
    e.g. get_solution_string(my_submission)
    """
    return (sub.installation_id + sub.accuracy_group.map(str)).str.cat()

def parse_solution_string(s: str) -> pd.DataFrame:
    row_length = 9  # 8 characters for installation id, 1 for result
    num_solutions = int(len(submission_str) / row_length)
    row_starts = range(0, (num_solutions+1)*row_length, row_length)
    rows = [(s[i:j-1], s[j-1:j]) for i, j in zip(row_starts[:-1], row_starts[1:])]
    return pd.DataFrame(rows, columns=['installation_id', 'accuracy_group'])


# In[ ]:


# Paste submission as string, should be exactly 9000 characters.
submission_str = '00abaee70012422180'


# In[ ]:


submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

df_predict = parse_solution_string(submission_str)

for i, row in df_predict.iterrows():
    submission.loc[submission['installation_id'] == row['installation_id'], 'accuracy_group'] = row['accuracy_group']
    
submission.to_csv('submission.csv', index=False)
submission.head()

