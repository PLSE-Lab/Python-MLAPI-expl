#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train_df = pd.read_csv("../input/train.csv")


# In[ ]:


kaggle=train_df.loc[train_df['question_text'].str.contains("(?i)kaggle")].copy().reset_index()


# In[ ]:


list(kaggle['question_text'])


# In[ ]:




