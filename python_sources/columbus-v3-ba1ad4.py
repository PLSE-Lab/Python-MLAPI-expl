#!/usr/bin/env python
# coding: utf-8

# # Getting insights from GA data #
# 
# 
# This uses a public bigquery dataset, specifically  [google_analytics_sample](https://bigquery.cloud.google.com/dataset/bigquery-public-data:google_analytics_sample). 
# 
# 
# 

# ## Fetch data from Bigquery ##
# 
# Run to see data structure for Bigquery to understand each columns.
# 

# In[ ]:


print('What tables do I have?')
import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
bq_assistant.list_tables()[:5]


# In[ ]:


QUERY = """
    SELECT 
        *  -- Warning, be careful when doing SELECT ALL
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 
    LIMIT 100
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


#  ## Get some useful data  ##

# **Visits on a particular day**
# Write and run a query to provide number of visits on 28/01/2017.

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
schema = bq_assistant.table_schema('ga_sessions_20170128')
QUERY = "SELECT SUM(totals.visits) as visits FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170128`"
df = bq_assistant.query_to_pandas(QUERY)
df.head()


# **Visits for a number of days** 
#    Write and run a query to provide number of visits between 01/01/2017 and 31/01/2017

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
QUERY = "SELECT SUM (totals.visits) as Visits FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` WHERE _TABLE_SUFFIX BETWEEN '20170101' AND '20170131'"
df = bq_assistant.query_to_pandas(QUERY)
df.head()


# **Visits by day for a number of days**
# Write and run a query to provide number of visits by day between 01/01/2017 and 31/01/2017

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
QUERY = """
    SELECT
        date as date,
        SUM(totals.visits) as visits
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY date ASC
    LIMIT 31
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(31)


# List daily revenue for Jan 2017, order by highest to lowest.

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
QUERY = """
    SELECT
        date as date,
        SUM(totals.transactionRevenue) as transactionRevenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY transactionRevenue DESC
    LIMIT 31
"""
df = bq_assistant.query_to_pandas(QUERY)
df.head(31)


# How much revenue was there in June 2017 

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
QUERY = """
    SELECT
        
        SUM(totals.transactionRevenue) as totalRevenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630' 
"""
df = bq_assistant.query_to_pandas(QUERY)
df.head(31)


# In June 2017, which traffic source has brought the maximum revenue?

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
schema = bq_assistant.table_schema('ga_sessions_20170630')
QUERY = """
    SELECT
        trafficSource.source as Source ,totals.transactionRevenue as Revenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630' 
"""
df = bq_assistant.query_to_pandas(QUERY)
df.head() 


# Pull through user journey for each user on 08/01/17

# In[ ]:


import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
schema = bq_assistant.table_schema('ga_sessions_20170630')
QUERY = """
  SELECT 
      fullVisitorId, 
      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,
      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,
      hits.page.pagePath,
      hits.type
  FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_20170108` AS ga, 
      unnest(hits) as hits
  ORDER BY hitTime 
  LIMIT 50
  """

df = bq_assistant.query_to_pandas(QUERY)
df.head() 

