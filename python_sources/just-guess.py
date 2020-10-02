#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train_human_labels.csv')
df.head()


# In[ ]:


l = df.LabelName.value_counts().index.tolist()[:4]


# In[ ]:


an = ' '.join(l)


# In[ ]:


print(l)


# In[ ]:


print(an)


# In[ ]:


df = pd.read_csv('../input/stage_1_sample_submission.csv', usecols=[0])


# In[ ]:


ans = []
for i in range(len(df)):
    ans.append(an)


# In[ ]:


df['labels'] = ans
df.head()


# In[ ]:


df.to_csv('sub.csv', index=False)

