#!/usr/bin/env python
# coding: utf-8

# This is some fun analysis of the Google Patents Public data set.
# 
# First, a picture of one of the more enjoyable patents of all time...lego patent [USD253711S](https://patents.google.com/patent/USD253711S/en) :)

# In[ ]:


from IPython.display import Image
url = 'https://patentimages.storage.googleapis.com/pages/USD253711-2.png'
Image(url=url,width=800, height=600)


# **Step 1** - Prep BigQuery connection.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
from bq_helper import BigQueryHelper
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

# prepare bigQuery helper
patents = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patents")
bq_assistant = BigQueryHelper("patents-public-data", "patents")
bq_assistant.list_tables()
#patents.table_schema("publications")
#patents.head("publications")
#patents.head("publications",selected_columns="application_number_formatted,inventor,country_code", num_rows=10)


# **Step 2** - Load USPTO patent codes that I brought in as a new dataset from the [USPTO website](https://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm).

# In[ ]:


#import os
#print(os.listdir("../input"))
USPTO_patent_codes = pd.read_csv('../input/USPTO patent codes.csv')
USPTO_patent_codes.head(5)


# **Step 3** - Query to find all inventors that have over 100 patents that were granted. I was happy that I got to use two of my favorite SQL commands in this: WITH and UNNEST. :)

# In[ ]:


# create query to be run
query1 = """
WITH temp1 AS (
    SELECT
      DISTINCT
      PUB.country_code,
      PUB.application_number AS patent_number,
      inventor_name
    FROM
      `patents-public-data.patents.publications` PUB
    CROSS JOIN
      UNNEST(PUB.inventor) AS inventor_name
    WHERE
          PUB.grant_date > 0
      AND PUB.country_code IS NOT NULL
      AND PUB.application_number IS NOT NULL
      AND PUB.inventor IS NOT NULL
)
SELECT
  *
FROM (
    SELECT
     temp1.country_code AS country,
     temp1.inventor_name AS inventor,
     COUNT(temp1.patent_number) AS count_of_patents
    FROM temp1
    GROUP BY
     temp1.country_code,
     temp1.inventor_name
     )
WHERE
 count_of_patents > 100
;
"""
# Check size of data being examined by query
#patents.estimate_query_size(query1)

# Run query 
query_results = patents.query_to_pandas_safe(query1, max_gb_scanned=6)
print("Number of records:", len(query_results.index))
query_results.head(5)


# **Step 4** - sort the results to view the top 50 inventors in the US by volume of patents. I had fun googling some of the top names on here to get a sense for what they're focused on.

# In[ ]:


# reduce results down to the top 50 inventors in the US
top_50_inventors = query_results.loc[query_results['country'] == "US"].nlargest(50,'count_of_patents')

# show the top 50 in a plotly table
table1 = ff.create_table(top_50_inventors)
py.iplot(table1, filename='jupyter-table1')


# **TO DO...Step 5** - Plot top areas of invention for the top 50 inventors in the US by bringing the Google patent data together with the USPTO codes.
