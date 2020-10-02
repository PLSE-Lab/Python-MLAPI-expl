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

from subprocess import check_output
# Any results you write to the current directory are saved as output.


# In[ ]:


import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                   dataset_name="openaq")

open_aq.list_tables()


# In[ ]:


open_aq.head("global_air_quality")


# In[ ]:


query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
         """


# In[ ]:


us_cities = open_aq.query_to_pandas_safe(query) 


# In[ ]:


us_cities.city.value_counts().head()


# In[ ]:


query = """SELECT DISTINCT (country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
         """


# In[ ]:


no_ppm_countries = open_aq.query_to_pandas_safe(query)


# In[ ]:


no_ppm_countries.shape
no_ppm_countries.head()


# In[ ]:


no_ppm_countries


# In[ ]:


query = """SELECT DISTINCT (pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
         """


# In[ ]:


no_pollutant = open_aq.query_to_pandas_safe(query)


# In[ ]:


no_pollutant.shape


# In[ ]:


no_pollutant


# In[ ]:




