#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper


# In[ ]:


air_data = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "epa_historical_air_quality")


# In[ ]:


l_tables = air_data.list_tables()


# In[ ]:


print (l_tables[3])
air_data.table_schema(l_tables[3])


# In[ ]:


air_data.head(l_tables[3], selected_columns=['site_num', 'arithmetic_mean'])


# In[ ]:


query = """SELECT arithmetic_mean
            FROM `bigquery-public-data.epa_historical_air_quality.hap_daily_summary`
            WHERE site_num = "0005" """

air_data.estimate_query_size(query)

