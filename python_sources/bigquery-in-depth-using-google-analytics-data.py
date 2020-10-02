#!/usr/bin/env python
# coding: utf-8

# # BigQuery: In depth using Google Analytics data
# Google Analytics team has shared the dataset for Google Merchandise Store. This data is same as of a typical ecommerce website. Let's see what BigQuery can do with this dataset.
# 
# Read introductory part here: https://www.kaggle.com/vikramtiwari/bigquery-getting-started-with-bq-data-on-kaggle

# In[ ]:


# offical google cloud library is recommended
from google.cloud import bigquery
import pandas as pd


# In[ ]:


# initiate bigquery client
bq = bigquery.Client()


# In[ ]:


# let's start with first table in the dataset
query = """
SELECT
    *
FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`
LIMIT 10
"""
# NOTE: Don't use SELECT * in your queries.


# In[ ]:


# make the query and get the result to a dataframe
result = bq.query(query).to_dataframe()


# In[ ]:


# let's see what this data contains
result


# We see that interesting part of data is in json format inside a single column. BigQuery provides ways to query on those specific keys as well.

# In[ ]:


keyed_query = """
SELECT
    device.browser as browser,
    SUM ( totals.visits ) AS total_visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`
GROUP BY
    device.browser
ORDER BY
    total_visits DESC
"""
keyed_df = bq.query(keyed_query).to_dataframe()
keyed_df


# To perform more complex queries you can also UNNEST the data and then perform your opeations. Note that you can also use multiple locations in FROM statement.

# In[ ]:


unnested_query = """
SELECT
    fullVisitorId,
    visitId,
    visitNumber,
    hits.hitNumber AS hitNumber,
    hits.page.pagePath AS pagePath
FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`, UNNEST(hits) as hits
WHERE
    hits.type="PAGE"
ORDER BY
    fullVisitorId,
    visitId,
    visitNumber,
    hitNumber
"""
unnested_query_df = bq.query(unnested_query).to_dataframe()
unnested_query_df


# An easier way to query on multiple batch of tables is to use table ranges while querying. This works best on tables that are partitoned using some set format (generally date). Note that you can also pass parameters as you would to any string in Python.

# In[ ]:


parameterized_query = """
SELECT
    device.browser as browser,
    SUM ( totals.visits ) AS total_visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
    _TABLE_SUFFIX BETWEEN '%s' AND '%s'
GROUP BY
    device.browser
ORDER BY
    total_visits DESC
"""
start_date = '20170101'
end_date = '20171231'
parameterized_query_df = bq.query((parameterized_query % (start_date, end_date))).to_dataframe()
parameterized_query_df


# When analysis becomes complex, you can also use sub-queries.

# In[ ]:


sub_query = """
SELECT
    ( SUM(total_transactionrevenue_per_user) / SUM(total_visits_per_user) ) AS avg_revenue_by_user_per_visit
FROM (
    SELECT
        fullVisitorId,
        SUM( totals.visits ) AS total_visits_per_user,
        SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '20170701' AND '20170731' AND totals.visits > 0
        AND totals.transactions >= 1 AND totals.transactionRevenue IS NOT NULL
    GROUP BY
        fullVisitorId)
"""
sub_query_df = bq.query(sub_query).to_dataframe()
sub_query_df


# And when even subqueries are not enough, you can create your own functions for more detailed manipulations. These functions are called User Definded Functions and can be implemented using JavaScript with BigQuery.

# In[ ]:


# NOTE: We are not using any existing data source in this example query, we are generating our data source and then using the result as data source.
udf_query = """
CREATE TEMPORARY FUNCTION multiplyInputs(x FLOAT64, y FLOAT64)
RETURNS FLOAT64
LANGUAGE js AS '''
  return x*y;
''';

CREATE TEMPORARY FUNCTION divideByTwo(x FLOAT64)
RETURNS FLOAT64
LANGUAGE js AS '''
  return x/2;
''';

WITH numbers AS
  (SELECT 1 AS x, 5 as y
  UNION ALL
  SELECT 2 AS x, 10 as y
  UNION ALL
  SELECT 3 as x, 15 as y)

SELECT x,
  y,
  multiplyInputs(divideByTwo(x), divideByTwo(y)) as half_product
FROM numbers;
"""
udf_df = bq.query(udf_query).to_dataframe()
udf_df


# Learn more about BigQuery architecture:
# - https://cloud.google.com/files/BigQueryTechnicalWP.pdf
# 
# BigQuery documentation and more samples:
# - https://cloud.google.com/bigquery/docs/reference/standard-sql/
# - https://medium.com/@hoffa/400-000-github-repositories-1-billion-files-14-terabytes-of-code-spaces-or-tabs-7cfe0b5dd7fd
# 
# If you are interested in BigQuery and it's capabilities, here are a few accounts you can follow to stay updated:
# - https://twitter.com/felipehoffa
# - https://twitter.com/ElliottBrossard
# - https://twitter.com/thetinot
# - https://twitter.com/jrdntgn
# - https://twitter.com/polleyg
# - https://twitter.com/GCPDataML
# 
# Find me on twitter:
# - https://twitter.com/Vikram_Tiwari

# > BigQuery is a very powerful data analytics plaform. What will you build with it?

# In[ ]:




