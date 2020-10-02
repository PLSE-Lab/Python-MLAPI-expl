#!/usr/bin/env python
# coding: utf-8

# # Import `bq_helper`

# In[ ]:


import bq_helper

# CREATE OBJECT 

baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")


# In[ ]:


baseball.list_tables() #TABLES REVIEW


# In[ ]:


baseball.head('games_wide') # TABLE PREVIEW


# 
# # ** Step by step to make a query: **
# 1. Declare the **`query`** with the appropriate syntax
# 2. We create an object to save the query 
# 3. Impression of our query

# In[ ]:


query = """SELECT homeTeamName,awayTeamName,homeFinalRuns
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE homeFinalRuns <10
             """


# In[ ]:


Number_team = baseball.query_to_pandas_safe(query)


# In[ ]:


Number_team


# # Next Query **`GROUP BY`**

# In[ ]:


query = """SELECT homeTeamName,venueMarket
            FROM `bigquery-public-data.baseball.games_wide`
            GROUP BY homeTeamName,venueMarket;
             """  


# In[ ]:


market_by_team = baseball.query_to_pandas_safe(query)


# In[ ]:


market_by_team


# **IMPORTANT :**
# 
# It is recommended to follow the syntax for the use of other functions in SQL, I hope you have served them
#     

# # **FINAL**

# In[ ]:




