#!/usr/bin/env python
# coding: utf-8

# This basic Python kernel shows you how to query the `commits` table in the GitHub Repos BigQuery dataset, look at the commit messages it returns, and create a word cloud.

# In[11]:


from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

QUERY = """
        SELECT message
        FROM `bigquery-public-data.github_repos.commits`
        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20
        LIMIT 2000
        """

query_job = client.query(QUERY)

iterator = query_job.result(timeout=30)
rows = list(iterator)

commit_messages = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
commit_messages.head(10)


# In[7]:


import wordcloud
import matplotlib.pyplot as plt

words = ' '.join(commit_messages.message).lower()
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=300,
                            relative_scaling=.5).generate(words)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('github-commit-message-wordcloud.png')
plt.imshow(cloud);


# If you fork and modify this query or write a new kernel of your own on this BigQuery dataset, make sure to follow [these best practices for resource usage](https://cloud.google.com/bigquery/docs/best-practices-costs) so that you don't exceed quotas.
