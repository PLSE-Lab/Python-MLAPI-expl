#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


from bq_helper import BigQueryHelper
gitrepos = BigQueryHelper("bigquery-public-data", "github_repos")


# In[ ]:


print(" List of Tables..")
pprint(gitrepos.list_tables())


# In[ ]:


test_query = """
            SELECT
                SPLIT(content, '\\n') AS line,
                id
            FROM
                `bigquery-public-data.github_repos.sample_contents`
            WHERE
                sample_path LIKE "%.py"
            """
gitrepos.estimate_query_size(test_query)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lines = gitrepos.query_to_pandas_safe(test_query, max_gb_scanned=24)')


# In[ ]:


# lines.head()

pprint(lines.line[0][10:30])


# In[ ]:


q_tab_or_space = ("""
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.py" )
SELECT
  Indentation,
  COUNT(Indentation) AS number_of_occurence
FROM (
  SELECT
    CASE
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^\t+")))>=1 THEN 'Tab'
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +")))>=1 THEN 'Space'
        ELSE 'Other'
    END AS Indentation
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^\s+")
  GROUP BY
    id )
GROUP BY
  Indentation
ORDER BY
  number_of_occurence DESC
""")

gitrepos.estimate_query_size(q_tab_or_space)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tab_space_df = gitrepos.query_to_pandas(q_tab_or_space)\ntab_space_df')


# In[ ]:


tab_space_df.plot.bar(x="Indentation",y="number_of_occurence", rot=0)


# In[ ]:


q_no_of_spaces = """
            #standardSQL
            WITH
              lines AS (
              SELECT
                SPLIT(content, '\\n') AS line,
                id
              FROM
                `bigquery-public-data.github_repos.sample_contents`
              WHERE
                sample_path LIKE "%.py" )
            SELECT
              space_count,
              COUNT(space_count) AS number_of_occurence
            FROM (
              SELECT
                id,
                MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +"))) AS space_count
              FROM
                lines
              CROSS JOIN
                UNNEST(lines.line) AS flatten_line
              WHERE
                REGEXP_CONTAINS(flatten_line, r"^ +")
              GROUP BY
                id )
            GROUP BY
              space_count
            ORDER BY
              number_of_occurence DESC
            """

gitrepos.estimate_query_size(q_no_of_spaces)


# In[ ]:


no_of_spaces_df = gitrepos.query_to_pandas(q_no_of_spaces)


# In[ ]:


no_of_spaces_df.head()


# In[ ]:


no_of_spaces_df[:6].plot.bar(x="space_count", y="number_of_occurence")

