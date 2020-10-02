# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

# Question #1 Countries that do not use ppm as unit
query = """SELECT DISTINCT country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm'"""
non_ppm_countries = open_aq.query_to_pandas_safe(query)

# Question #2 Pollutants that have a value of zero
query = """SELECT DISTINCT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0.00"""
zero_pollutants = open_aq.query_to_pandas_safe(query)
print(zero_pollutants)

