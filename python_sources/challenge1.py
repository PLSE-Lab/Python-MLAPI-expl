#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


import bq_helper


# In[ ]:


openAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")


# In[ ]:


openAQ.list_tables()


# In[ ]:


openAQ.table_schema("global_air_quality")


# In[ ]:


#Which countries use a unit other than ppm to measure any type of pollution? 


# In[ ]:


openAQ.head("global_air_quality")


# In[ ]:


query = """SELECT distinct country, unit FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm" 
        """ 


# In[ ]:


openAQ.estimate_query_size(query)


# In[ ]:


openAQ.query_to_pandas_safe(query, max_gb_scanned=0.1)


# In[ ]:


#Which pollutants have a value of exactly 0?


# In[ ]:


query = """SELECT distinct pollutant FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0 
        """ 


# In[ ]:


openAQ.estimate_query_size(query)


# In[ ]:


openAQ.query_to_pandas_safe(query, max_gb_scanned=0.1)


# In[ ]:




