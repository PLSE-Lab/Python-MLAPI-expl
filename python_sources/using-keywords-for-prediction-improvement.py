#!/usr/bin/env python
# coding: utf-8

# I want to do some postprocessing on output of this kernel:
# https://www.kaggle.com/kpriyanshu256/bert-fastai
# which is written by Priyanshu Kumar.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sub_Priyanshu = pd.read_csv("/kaggle/input/bert-fastai/submission.csv")

train.head()


# Priyanshu's submission is in *sub_Priyanshu* and I want to improve its score (0.82413) by knowledge in keywords.
# Let's look at the kewords with high probability of being disaster.

# In[ ]:


train = train.fillna('None')
ag = train.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})
ag.sort_values('Disaster Probability', ascending=False).head(10)


# It seems that some keywords almost always mention a real disaster.

# In[ ]:


keyword_list = list(ag[(ag['Count']>2) & (ag['Disaster Probability']>=0.9)].index)
keyword_list


# So I will find them in test set and replace corresponding targets by 1.

# In[ ]:


ids = test['id'][test.keyword.isin(keyword_list)].values
sub_Priyanshu['target'][sub_Priyanshu['id'].isin(ids)] = 1
sub_Priyanshu.head()


# In[ ]:


sub_Priyanshu.to_csv('sub_Priyanshu_modified.csv', index=False)

