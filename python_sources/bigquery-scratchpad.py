#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()


# In[ ]:


dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')
dataset = client.get_dataset(dataset_ref)


# In[ ]:


tables = list(client.list_tables(dataset))

for table in tables:
    print(table.table_id)


# In[ ]:


table_ref = dataset_ref.table('full')
table = client.get_table(table_ref)


# In[ ]:


table.schema


# In[ ]:


table_ref = dataset_ref.table('comments')
table = client.get_table(table_ref)


# In[ ]:


table.schema


# In[ ]:


client.list_rows(table, max_results=5).to_dataframe()

