#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 list |grep pandas')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -U pip')
get_ipython().run_line_magic('pip', 'install pandas==0.24.0')


# In[ ]:


get_ipython().system('pip3 list |grep pandas')


# In[ ]:


import pandas as pd
pd.__version__


# In[ ]:


idx = pd.period_range('2000', periods=4)
idx.array


# ### BigQuery

# In[ ]:


from google.cloud import bigquery


# In[ ]:


client = bigquery.Client()


# In[ ]:


hn_dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')


# In[ ]:


type(hn_dataset_ref)


# In[ ]:


hn_dset = client.get_dataset(hn_dataset_ref)

