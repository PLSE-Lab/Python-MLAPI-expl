#!/usr/bin/env python
# coding: utf-8

# **How to Query the Stack Overflow Data (BigQuery Dataset)**

# In[2]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")


# 

# In[3]:


bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()


# In[9]:


tags = bq_assistant.head("stackoverflow_posts", num_rows=5000000, selected_columns=['id', 'title', 'body', 'tags'])


# In[14]:


query1 = """SELECT
  COUNT(tags)
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
WHERE
    tags LIKE "%|%|%|%"
"""
tags_c = stackOverflow.query_to_pandas_safe(query1)
tags_c


# In[24]:


query = """SELECT
    sp.tags
FROM
  `bigquery-public-data.stackoverflow.stackoverflow_posts` sp
WHERE
    tags IS NOT NULL
LIMIT 1000000
"""
tags_only = stackOverflow.query_to_pandas_safe(query)


# 

# In[20]:


tags.info()


# In[21]:


tags.to_csv('stackoverflow_posts_5kk.csv.gz', compression='gzip', index=False)


# In[28]:


tags_only.info()
tags.to_csv('tags_only_1kk.csv.gz', compression='gzip', index=False)


# In[29]:


get_ipython().system(' ls -lah')


# In[30]:


from IPython.display import FileLinks
FileLinks('.') # input argument is specified folder


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "stackoverflow_posts_100k.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
# df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(tags)


# In[31]:


from collections import Counter, defaultdict


# In[ ]:


tags_only.tags.head().empty()


# In[ ]:


# skills_counter = Counter(tags_only.tags.str.lower().str.split("|").apply(list).sum())


# In[ ]:





# In[ ]:




