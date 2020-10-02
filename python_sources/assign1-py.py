# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")
query = """SELECT distinct country
            FROM 'bigquery-public-data.openaq.global_air_quality'
            WHERE unit != 'ppm'
        """
non_ppm_countries = open_aq.query_to_pandas_safe(query)
non_ppm_countries.head()
query1 = """SELECT distinct pollutant,value
            FROM 'bigquery-public-data.openaq.global_air_quality'
            WHERE value = 0.0
        """
non_pollutant_values = open_aq.query_to_pandas_safe(query1)
non_pollutant_values.head()