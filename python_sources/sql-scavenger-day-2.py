import pandas as pd
from google.cloud import bigquery
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

#hacker_news.list_tables()
#hacker_news.head("full")

#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query1="""SELECT type,count(distinct id) 
         FROM `bigquery-public-data.hacker_news.full`
         group by type"""
hacker_news.estimate_query_size(query1)
hacker_news.query_to_pandas(query1)

#How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
#hacker_news.head("comments")
query2="""SELECT count(distinct id) 
         FROM `bigquery-public-data.hacker_news.comments`
         where deleted=True
         """
hacker_news.estimate_query_size(query2)
hacker_news.query_to_pandas(query2)
