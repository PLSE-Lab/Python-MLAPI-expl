#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I copy and paste from https://www.kaggle.com/mrisdal/safely-analyzing-github-projects-popular-licenses

import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.list_tables()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.table_schema("licenses")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.head("licenses", num_rows=10)')


# In[ ]:


QUERY = """
        SELECT message
        FROM `bigquery-public-data.github_repos.commits`
        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20
        LIMIT 2000
        """
bq_assistant.estimate_query_size(QUERY)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.estimate_query_size(QUERY)')


# In[ ]:


QUERY = """
        SELECT message
        FROM `bigquery-public-data.github_repos.commits`
        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20
        LIMIT 2000
        """


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = bq_assistant.query_to_pandas_safe(QUERY)')


# In[ ]:


QUERY = """
        SELECT license, COUNT(*) AS count
        FROM `bigquery-public-data.github_repos.licenses`
        GROUP BY license
        ORDER BY COUNT(*) DESC
        """


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bq_assistant.estimate_query_size(QUERY)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = bq_assistant.query_to_pandas_safe(QUERY)')


# In[ ]:


print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

f, g = plt.subplots(figsize=(12, 9))
g = sns.barplot(x="license", y="count", data=df, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title("Popularity of Licenses Used by Open Source Projects on GitHub")
plt.show(g)


# In[ ]:




