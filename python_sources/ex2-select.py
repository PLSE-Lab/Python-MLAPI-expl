#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset('openaq', project='bigquery-public-data')
dataset = client.get_dataset(dataset_ref)


# In[ ]:


tables = list(client.list_tables(dataset))


# In[ ]:


for table in tables:
    print(table.table_id)


# In[ ]:


table_ref = dataset_ref.table('global_air_quality')
table = client.get_table(table_ref)


# In[ ]:


type(table)


# In[ ]:


client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


query = """
    SELECT city
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE country = 'US'
"""


# In[ ]:


query


# In[ ]:


query_job = client.query(query)


# In[ ]:


us_cities = query_job.to_dataframe()


# In[ ]:


type(us_cities)


# In[ ]:


us_cities.city.value_counts().head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'query_1 = """\n    SELECT city, country, source_name\n    FROM `bigquery-public-data.openaq.global_air_quality`\n    WHERE country = \'US\'\n"""\nquery_job_1 = client.query(query_1)\ndf_1 = query_job_1.to_dataframe()\ndf_1.head()')


# In[ ]:


# Only run the query if it's less than 1 MB
ONE_MB = 1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)
safe_query_job = client.query(query, job_config=safe_config)
safe_query_job.to_dataframe().head()


# In[ ]:




