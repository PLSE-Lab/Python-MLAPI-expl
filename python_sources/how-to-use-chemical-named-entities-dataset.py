#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("../input/training.csv")
df.Token.fillna('NA', inplace=True)
# Few values in original corpora is written as NA which is handled as nan in Pandas
# Same goes with validation.csv and evaluation.csv


# In[ ]:


for name, group in df.groupby(['PMID_Type', 'Sentence_Index']):
    print(group.Token)
    break


# In[ ]:


for name, group in df.groupby(['PMID_Type', 'Sentence_Index']):
    for idx, row in group.iterrows():
        print(row)
        break
    break
# Good luck for your exploration!


# In[ ]:




