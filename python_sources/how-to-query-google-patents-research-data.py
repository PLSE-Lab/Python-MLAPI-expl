#!/usr/bin/env python
# coding: utf-8

# # How to Query Google Patents Research Data (BigQuery)
# [Click here](https://www.kaggle.com/mrisdal/safely-analyzing-github-projects-popular-licenses) for a detailed notebook demonstrating how to use the bq_helper module and best practises for interacting with BigQuery datasets.

# In[ ]:


# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

from google.cloud import bigquery
import pandas as pd
patents_research = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="google_patents_research")


# In[ ]:


# View table names under the google_patents_research data table
bq_assistant = BigQueryHelper("patents-public-data", "google_patents_research")
bq_assistant.list_tables()


# In[ ]:


# View the first three rows of the publications data table
bq_assistant.head("publications", num_rows=3)


# In[ ]:


# View the last 3 rows of the publications data table
# bq_assistant.last("publications", num_rows=3) ##no such command? 


# In[ ]:


# View information on all columns in the trials data table
bq_assistant.table_schema("publications")


# ## Example SQL Query
# What countries do some of these patents belong to?

# In[ ]:


query_count = """
SELECT count(*)
FROM
  `patents-public-data.google_patents_research.publications`
"""
print(bq_assistant.estimate_query_size(query_count))
patents_research.query_to_pandas_safe(query_count)

### 121 million rows/patents


# In[ ]:


# query_country = """
# SELECT DISTINCT
#   country
# FROM
#   `patents-public-data.google_patents_research.publications`
# LIMIT
#   500;
#         """
# query_country = patents_research.query_to_pandas_safe(query_country, max_gb_scanned=25)
# query_country

#### 'United States' , 'Eurasian Patent Office' , 'United Kingdom' , 'WIPO (PCT)' , 'EUIPO' , 'USSR - Soviet Union' ...


# #### get the top terms (and cpcs), for publications which have any
# * queries about 45 GB. 
# 
# * OFFSEt ->  skip the first X million with offset  . (data seems to be ordered by date, despite filing date not being in this table)
# 
# * could combine with full patent table to get filing date here.. 
#     * https://www.kaggle.com/dhimananubhav/china-2-million-patents-for-invention#Publications-from-China-%F0%9F%87%A8%F0%9F%87%B3

# In[ ]:


query1 = """
SELECT 
  publication_number, top_terms, title
FROM
  `patents-public-data.google_patents_research.publications`
WHERE
(ARRAY_LENGTH(top_terms)> 0) AND (title_translated = FALSE) 
AND (CHAR_LENGTH(title)>2)
AND (country = "United States" OR country = "USA")
AND (publication_description LIKE "Patent%")

LIMIT 4123456 OFFSET 12123456
;
        """

# cpc, ## cpc.code causes error, and returning the cpc seems to cause memory issues + be much slower for some reason  
##   (LEN(top_terms)>1)
# (NOT isnull(title))
# AND (publication_description LIKE "%atent%")

# publication_description - Patent , [china :  Granted Patent , Granted patent for invention ..  ]

#  LIMIT
#    27123456

## could filter by last letter(2) of publication_number , corresponds to patent kind / type. (e.g. A = patent, granted)

print(bq_assistant.estimate_query_size(query1))


# * USA patent publication_description :
# * Patent application publication                                       1094262
# * Patent                                                               1002229
# * Patent ( having previously published pre-grant publication)           554164
# * Patent ( no pre-grant publication)                                    131266
# * Design patent                                                         112373
# * Reissue patent                                                          5675
# * Plant patent ( no pre-grant publication)                                2233
# * Patent application publication ( corrected publication)                 1586
# * Plant patent application publication                                    1525
# * Plant patent                                                            1243
# * Plant patent ( having previously published pre-grant publication)       1078
# * Patent application publication ( republication)                          326

# In[ ]:


df = patents_research.query_to_pandas_safe(query1, max_gb_scanned=50)

df.head(20) ## with an offset of 13123456 , and no special other filtering, we see the first patent in 2012


# In[ ]:


# response1 = df
print(df.shape)


# In[ ]:


df.tail(30)


# In[ ]:


# print(df["publication_description"].value_counts())


# ### export
# *If we have cpc, we'd also want to export the cpc codes
# * kernels have limit on file size

# In[ ]:


print(df.shape[0])
# df = response1.drop(["title_translated","country"],axis=1,errors="ignore")
df = df.drop_duplicates(subset=["publication_number","title"])
print(df.shape[0])
df


# In[ ]:


df.to_csv("sample_usa_patents_research_k_bq_terms_last.csv.gz",index=False,compression="gzip")


# ##### analyze the top terms

# In[ ]:


# response1


# In[ ]:


### https://www.kaggle.com/shaileshshettyd/china-patents-contributions
top_terms = pd.DataFrame(df["top_terms"].tolist())

top_terms = pd.DataFrame(top_terms.values.flatten())
top_terms.columns = ['top_terms']

top_terms = top_terms.dropna(axis=0,how='all')
top_terms.shape


# In[ ]:


# 9 millions terms
top_terms.sample(5)


# In[ ]:


#  unique terms
top_terms.top_terms.nunique()


# ### Frequently occuring salient terms

# In[ ]:


df_agg = pd.DataFrame(top_terms.sample(frac=0.02).groupby('top_terms')['top_terms'].count())

df_agg.columns = ['counter']
df_agg = df_agg.sort_values('counter', ascending=False)
# df_agg = df_agg.head(30)

# df_agg.tail(5)

df_agg.head(50)


# In[ ]:


df_agg.tail(30)


# 

# In[ ]:


# # from pandas import json_normalize
# from pandas.io.json import json_normalize
def get_nested_codes(row):
    ls = []
#     row = json_normalize(row)
    for i in row:
        ls.append(i["code"])
    return(ls)


# In[ ]:


# print(df["cpc"][0])
# df["cpc"] = df["cpc"].apply(get_nested_codes)
# print(df["cpc"][0])


# ## Importance of Knowing Your Query Sizes
# 
# It is important to understand how much data is being scanned in each query due to the free 5TB per month quota. For example, if a query is formed that scans all of the data in a particular column, given how large BigQuery datasets are it wouldn't be too surprising if it burns through a large chunk of that monthly quota!
# 
# Fortunately, the bq_helper module gives us tools to very easily estimate the size of our queries before running a query. Start by drafting up a query using BigQuery's Standard SQL syntax. Next, call the estimate_query_size function which will return the size of the query in GB. That way you can get a sense of how much data is being scanned before actually running your query.
