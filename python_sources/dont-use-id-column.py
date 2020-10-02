#!/usr/bin/env python
# coding: utf-8

# # Data Error
#  - Mismatch : Question IDs and Questions Full Text 
#  - Around 27K questions have this mismatch
#  - Shldnt be a big issue, since we always tend to drop this column, But highlighting it nonetheless

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df=pd.read_csv('/kaggle/input/question-pairs-dataset/questions.csv')
df.shape


# ### Drop NA rows

# In[ ]:


df.isna().sum()


# In[ ]:


df.dropna(axis='rows',inplace=True)
df.shape


# ### Create all Question-ids to Questions mapping

# In[ ]:


df1=df[['qid1','question1']].rename(columns={'qid1':'qid','question1':'question'})
df2=df[['qid2','question2']].rename(columns={'qid2':'qid','question2':'question'})


# In[ ]:


df_q1q2=pd.concat([df1,df2],axis=0)
df_q1q2.shape


# In[ ]:


df_q1q2.drop_duplicates(inplace=True)
df_q1q2.shape


# In[ ]:


df_q1q2.head()


# ### Non-Unique qid to question mappings
#  - For each qid, the content should be unique

# In[ ]:


df_wrongmappings=df_q1q2.groupby('qid').size()[df_q1q2.groupby('qid').size()>1].to_frame().reset_index()


# In[ ]:


df_q1q2_wrong=df_wrongmappings[['qid']].merge(df_q1q2,how='inner',on='qid')


# In[ ]:


df_q1q2_wrong.head(100)


# In[ ]:


df_q1q2_wrong.shape


# In[ ]:




