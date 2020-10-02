# Import the required libraries 

from bq_helper import BigQueryHelper
import bq_helper

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

medicare = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="cms_medicare")
bq_assistant = BigQueryHelper("bigquery-public-data", "cms_medicare")

query11 = """SELECT *
FROM
  `bigquery-public-data.cms_medicare.inpatient_charges_2011`;"""
response11 = medicare.query_to_pandas_safe(query11)
response11['year'] = 2011
response11.columns

query12 = """SELECT *
FROM
  `bigquery-public-data.cms_medicare.inpatient_charges_2012`;"""
response12 = medicare.query_to_pandas_safe(query12)
response12['year'] = 2012

query13 = """SELECT *
FROM
  `bigquery-public-data.cms_medicare.inpatient_charges_2013`;"""
response13 = medicare.query_to_pandas_safe(query13)
response13['year'] = 2013

query14 = """SELECT *
FROM
  `bigquery-public-data.cms_medicare.inpatient_charges_2014`;"""
response14 = medicare.query_to_pandas_safe(query14)
response14['year'] = 2014

query15 = """SELECT *
FROM
  `bigquery-public-data.cms_medicare.inpatient_charges_2015`;"""
response15 = medicare.query_to_pandas_safe(query15)
response15['year'] = 2015

full = pd.concat([response11, response12, response13, response14, response15] )
full.year.value_counts()

print(full.total_discharges.sum())
print(full.groupby('year')['total_discharges'].sum())

full.to_csv('data_cms.csv',index=False)