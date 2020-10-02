# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
                                       
query = """SELECT count (distinct id) As unique_ids
            FROM `bigquery-public-data.hacker_news.full`
             """

Unique_id = hacker_news.query_to_pandas_safe(query)


# save our dataframe as a .csv 
Unique_id.to_csv("Unique_id.csv")

query2 = """SELECT count (id) as Id_count, deleted
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type='comment' 
            GROUP BY deleted
             """
 # why do I get an error if I add AND deleted='True' to where clasue?            
deleted_comments=hacker_news.query_to_pandas_safe(query2)

deleted_comments.to_csv ("deleted.csv")
