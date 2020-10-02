#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import pandas as pd


# In[ ]:


# Create a "Client" object
client = bigquery.Client()


# In[ ]:


# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# In[ ]:


# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)


# In[ ]:


# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)


# In[ ]:


# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema


# In[ ]:


# Preview the first five lines of the "full" table
sample_file=client.list_rows(table, max_results=20).to_dataframe()
sample_file


# In[ ]:


sample_file.to_csv('twenty_entry.csv',encoding='utf-8', index=False)


# In[ ]:


# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1], max_results=20).to_dataframe()


# In[ ]:


# link to all the BigQuery client commands
# https://googleapis.github.io/google-cloud-python/latest/bigquery/reference.html


# ### Mentions of Kaggle on Hacker News

# In[ ]:


# Using WHERE reduces the amount of data scanned / quota used
query = """
SELECT title, time_ts
FROM `bigquery-public-data.hacker_news.stories`
WHERE REGEXP_CONTAINS(title, r"(k|K)aggle")
ORDER BY time
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
headlines.head(10)


# ###  Now let's create a word cloud

# In[ ]:


import wordcloud
import matplotlib.pyplot as plt

words = ' '.join(headlines.title).lower()
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=300,
                            relative_scaling=.5).generate(words)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('kaggle-hackernews.png')
plt.imshow(cloud)
plt.show()


# ### Reference : [Mentions of Kaggle on Hacker News](https://www.kaggle.com/mrisdal/mentions-of-kaggle-on-hacker-news)

# In[ ]:




