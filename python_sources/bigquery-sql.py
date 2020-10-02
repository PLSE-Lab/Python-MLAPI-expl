#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import pandas as pd


# In[ ]:


client = bigquery.Client()


# In[ ]:


dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)


# In[ ]:


tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)


# In[ ]:


table_ref = dataset_ref.table("comments")
table = client.get_table(table_ref)


# In[ ]:


table.schema


# In[ ]:


bq_file = client.list_rows(table, max_results=30).to_dataframe()


# In[ ]:


client.list_rows(table, selected_fields=table.schema[:5], max_results=5).to_dataframe()


# In[ ]:


#save file
with open("my_file.csv", "w") as file:
    file.write(bq_file.to_csv())


# In[ ]:




