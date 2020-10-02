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

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# List all tables in openaq schema
open_aq.list_tables()

#list first 5 rows of the table global_air_quality
open_aq.head("global_air_quality")

# Day1 (SQL hunt) question Which countries use a unit 
#other than ppm to measure any type of pollution?
#(Hint: to get rows where the value isn't something, use "!=")
#-- sql comment

Qry_countryunit_notppm = """
            SELECT distinct country --,city,unit,pollutant,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
Pd_unitnoppm = open_aq.query_to_pandas_safe(Qry_countryunit_notppm)

#print total number of countries use unit other than ppm
print("Total countries unit other than ppm is - ", str(Pd_unitnoppm.size))

#list first 5 countries uses unit other than ppm
print(" First 5 Countries  unit other than ppm - \n", str(Pd_unitnoppm.country.head()))
#open_aq.estimate_query_size(countryunit_notppm_query)

# Day1 (SQL hunt) question :Which pollutants have a value of exactly 0?
Qry_poll_val0 ="""
            SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
        """
#load into panda safe mode
pd_poll_val0 = open_aq.query_to_pandas_safe(Qry_poll_val0)

#print number of pollutants value is zero
print("Total pollutants  value exactly zero is  - ", str(pd_poll_val0.size))


print("First 5 Pollutants values zero is - \n", pd_poll_val0)