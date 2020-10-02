#!/usr/bin/env python
# coding: utf-8

# *Step 1: Identify labeled dataset*

# In[13]:


import numpy as np
import pandas as pd 
import os
#print(os.listdir("../input/"))
diabetes = pd.read_csv('../input/scratchpad/diabetes.csv')
diabetes['Id'] = diabetes.index
diabetes.head()


# *Step 2: Split dataset into training data and testing data*

# In[14]:


diabetes['split'] = np.random.randn(diabetes.shape[0], 1)
msk = np.random.rand(len(diabetes)) <= 0.7
train = diabetes[msk]
test = diabetes[~msk]

train = train.drop(columns='split')
train.head()


# In[15]:


test2 = test.drop(columns='Outcome')
test2 = test2.drop(columns='split')
test2.head()


# *Step 3: Create Answer Key and Sample Submission File*

# In[16]:


answer_key = test['Outcome']
answer_key = pd.DataFrame(answer_key)
answer_key['Id'] = answer_key.index
answer_key = answer_key[['Id', 'Outcome']]
answer_key.head()


# In[17]:


sample_submission = test['Outcome']
sample_submission = sample_submission.replace(0)
sample_submission = pd.DataFrame(sample_submission)
sample_submission['Id'] = sample_submission.index
sample_submission = sample_submission[['Id', 'Outcome']]
sample_submission.head()


# *Step 4: Save new files as CSV files*

# In[18]:


train.to_csv('train.csv', index=False)
test2.to_csv('test.csv', index=False)
answer_key.to_csv('answer_key.csv', index=False)
sample_submission.to_csv('sample_submission.csv', index=False)


# *Step 5: Setup Kaggle InClass Competition*

# 1. Navigate to http://www.kaggle.com/inclass
# 1. Follow the [instructions](https://www.kaggle.com/about/inclass/how-it-works) to setup an InClass competition
# 1. Upload files train.csv, test.csv, answer_key.csv, and sample_submission.csv when prompted
