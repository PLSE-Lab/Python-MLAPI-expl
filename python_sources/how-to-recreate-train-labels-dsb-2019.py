#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # DSB Assessment Grades

# ## Objective: Reconcile train and train_labels

# In trying to understand the requirements of the competition, I created a kernel to reconcile the event data in `train.csv` with the assessment accuracy groups in `train_labels.csv`.  In doing so, I have also extracted a corresponding `test_labels.csv` for the graded assessments found in the `test` split.  These data could be potentially used to augment the training set.

# ## Background

# As stated in the data description, there are four accuracy groups:
# > 3: the assessment was solved on the first attempt
# >
# > 2: the assessment was solved on the second attempt
# >
# > 1: the assessment was solved after 3 or more attempts
# >
# > 0: the assessment was never solved
# 
# In the event data of `train.csv`, we are told  that the number of attempts correspond to the 4100 event code for four of the assessment types and 4110 for one of the event types.  Within the `event_data` field, there is a flag for whether the attempt was correct or not.

# ## Procedure

# As such, we should be able to recreate the `train_labels.csv` file by:
# 1. Extracting all of the assessment scoring events (4100 or 4110 depending on type).
# 2. Counting the number of correct and incorrect attempts by `installation_id` and `game_session`
# 3. Applying a heuristic on the attempt counts to create the accuracy group.

# In[ ]:


import pandas as pd

def extract_accuracy_group(df: pd.DataFrame) -> pd.DataFrame:
    # Regex strings for matching Assessment Types
    assessment_4100 = '|'.join(['Mushroom Sorter',
                                'Chest Sorter',
                                'Cauldron Filler',
                                'Cart Balancer'])
    assessment_4110 = 'Bird Measurer'
    
    # 1. Extract all assessment scoring events
    score_events = df[((df['title'].str.contains(assessment_4110)) & (df['event_code']==4110)) |                      ((df['title'].str.contains(assessment_4100)) & (df['event_code']==4100))]
    
    # 2. Count number of correct vs. attempts
    # 2.a. Create flags for correct vs incorrect
    score_events['correct'] = 1
    score_events['correct'] = score_events['correct'].where(score_events['event_data'].str.contains('"correct":true'),other=0)
    
    score_events['incorrect'] = 1
    score_events['incorrect'] = score_events['incorrect'].where(score_events['event_data'].str.contains('"correct":false'),other=0)
    
    # 2.b. Aggregate by `installation_id`,`game_session`,`title`
    score_events_sum = score_events.groupby(['installation_id','game_session','title'])['correct','incorrect'].sum()
    
    # 3. Apply heuristic to convert counts into accuracy group
    # 3.a. Define heuristic
    def acc_group(row: pd.Series) -> int:
        if row['correct'] == 0:
            return 0
        elif row['incorrect'] == 0:
            return 3
        elif row['incorrect'] == 1:
            return 2
        else:
            return 1
        
    # 3.b. Apply heuristic to count data
    score_events_sum['accuracy_group'] = score_events_sum.apply(acc_group,axis=1)
    
    return score_events_sum


# ## Results

# In[ ]:


import os

DATA_DIR = '/kaggle/input/data-science-bowl-2019'


# ### Train Reconciliation

# In[ ]:


# Read `train.csv`
train = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))


# In[ ]:


# Run reconciliation
train_labels_extracted = extract_accuracy_group(train)


# #### Sample first rows

# First, we can eyeball the top several rows to see if we are in the ballpark.

# In[ ]:


train_labels_extracted.head(20)


# In[ ]:


# Read `train_labels.csv`
train_labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))


# In[ ]:


train_labels.drop(['accuracy'], axis=1).head(20)


# #### Compare Counts

# The distribution of accuracy groups appears to match between the two.

# In[ ]:


train_labels_extracted['accuracy_group'].value_counts()


# In[ ]:


train_labels['accuracy_group'].value_counts()


# #### Ensure matching `game_session`s

# In[ ]:


# Flatten multi-index
train_labels_extracted.reset_index(inplace=True)
train_labels_extracted.head()


# In[ ]:


extracted_train_sessions = set(train_labels_extracted['game_session'])
train_sessions = set(train_labels['game_session'])


# In[ ]:


extracted_train_sessions.symmetric_difference(train_sessions)


# Since the symmetric difference of the sets of the session ids is empty, then the session ids must match exactly between the two sets.

# #### Check `accuracy_group` column

# In[ ]:


extracted_train_groups = list(train_labels_extracted['accuracy_group'])
train_groups = list(train_labels['accuracy_group'])


# In[ ]:


all_match = True
for extract, gold in zip(extracted_train_groups, train_groups):
    if extract != gold:
        all_match = False
        break

if(all_match):
    print(f"All {len(extracted_train_groups)} groups match")
else:
    print(f"Found at least one mismatched group")


# ### Extracting labels from test set

# Since all of our tests pass, we should feel comfortable that our function is able to extract the accuracy score labels from the event data.  As such, this method should be able to extract scored assessments from the event histories in the test set as well.  By subsampling the event histories, we may be able to create more "training" examples that potentially correlate better with the user behavior profiles of the test set.

# In[ ]:


test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))


# In[ ]:


test_labels = extract_accuracy_group(test)


# In[ ]:


test_labels['accuracy_group'].value_counts()


# We have roughly 2,000 more labeled assessments to incorporate in our training procedures in `test_labels.csv`.

# In[ ]:


test_labels.to_csv('test_labels.csv')

