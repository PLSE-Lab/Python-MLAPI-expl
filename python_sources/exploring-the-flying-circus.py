#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sqlite3
import nltk

conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql(con=conn, sql='select * from scripts')
df['record_date'] = pd.to_datetime(df['record_date'])
df['transmission_date'] = pd.to_datetime(df['transmission_date'])
# Data needs some cleaning in a few places too
df['actor'] = df['actor'].str.replace('\n', ' ')
df.head()


# In[ ]:


# Who had the most lines?
df[df['type']=='Dialogue']['actor'].value_counts()


# In[ ]:


# Which character had the most lines?
df[df['type']=='Dialogue']['character'].value_counts().head(20)


# In[ ]:


# Which segment had the most lines of dialog?
df[df['type']=='Dialogue'].groupby(['segment'])['index'].count()    .reset_index(name='count').sort_values(['count'], ascending=False).head(20)


# In[ ]:


# How many segments are there in total?
len(df['segment'].unique())


# In[ ]:


#  which was the most prolific actor?
# person who appeared in the largest number of segments
df.groupby(['actor', 'segment'])['actor'].unique().value_counts()

