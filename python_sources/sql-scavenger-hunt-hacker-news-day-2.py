# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bh

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

hn = bh.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "hacker_news")

hn.list_tables()
hn.table_schema("full")
hn.head("full")

# Query to find out number of unique stories in the "full" table of hacker_news dataset
query1 = """ SELECT type, COUNT(id) AS NumberofUniqueStories FROM `bigquery-public-data.hacker_news.full` GROUP BY type """
query1_size = hn.estimate_query_size(query1)
query1_size
distinct_stories = hn.query_to_pandas(query1)
type(distinct_stories)
len(distinct_stories)
distinct_stories

#Query to find out the total number of comments that were deleted from the "full" table
query2 = """ SELECT deleted AS Deleted, COUNT(id) AS Numberofdeletedcomments FROM `bigquery-public-data.hacker_news.comments` GROUP BY deleted HAVING deleted = True """
query2_size = hn.estimate_query_size(query2)
query2_size
deleted_comments = hn.query_to_pandas(query2)

#Displaying the results
print("Break-up of Distinct Stories by Type")
print(distinct_stories)
print('\n')

print("Deleted Comments")
print(deleted_comments)
print("There are a total of", deleted_comments.iloc[0,1], "deleted comments in the 'comments' table of hacker_news dataset.")
# Writing results to file.
distinct_stories.to_csv("HackerNews_distinct_stories")
deleted_comments.to_csv("HackerNews_deleted_comments")

