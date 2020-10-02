#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import covid19_tools as cv19


# In[ ]:


df = pd.read_csv('../input/covid19-validation-set/covid_validation_set.csv')


# In[ ]:


df['Date'] = pd.to_datetime(df.Date)


# In[ ]:


df.head()


# In[ ]:


meta = cv19.load_metadata('../input/CORD-19-research-challenge/metadata.csv')


# In[ ]:


meta, covid19_counts = cv19.add_tag_covid19(meta)


# In[ ]:


val_df = df.merge(meta, left_on='URL', right_on='url', how='inner')


# In[ ]:


val_df.shape


# In[ ]:


val_df[['Title', 'Date', 'URL', 'tag_disease_covid19']].to_csv('andy_validation_results.csv')

