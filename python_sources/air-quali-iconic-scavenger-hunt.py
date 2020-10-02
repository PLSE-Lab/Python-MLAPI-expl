#!/usr/bin/env python
# coding: utf-8

# create a data object. (borrowed from Rachael)

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# first the countries with other units than ppm

# In[ ]:


# find countries that use units other than ppm
query1 = """ SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

country_otherunit = open_aq.query_to_pandas_safe(query1)

country_otherunit


# now the units that have values equal 0
# 

# In[ ]:


# find pollutants with value 0
query2 = """ SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

pollutant_val_0 = open_aq.query_to_pandas_safe(query2)

pollutant_val_0

