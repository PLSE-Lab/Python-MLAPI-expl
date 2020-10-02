# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bh # Importing Big Query Library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

open_aq = bh.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "openaq")

open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")

query1 = """SELECT country, unit FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm' """
query1_size = open_aq.estimate_query_size(query1)
unit_not_ppm = open_aq.query_to_pandas(query1)

print(unit_not_ppm)

query2 = """SELECT pollutant, value FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0 """
open_aq.estimate_query_size(query2)
pollutant_zero = open_aq.query_to_pandas(query2)

print(unit_not_ppm)
print(pollutant_zero)

unit_not_ppm.to_csv("openaq_countries_not_ppm")
pollutant_zero.to_csv("openaq_pollutant_value_zero")

# Any results you write to the current directory are saved as output.