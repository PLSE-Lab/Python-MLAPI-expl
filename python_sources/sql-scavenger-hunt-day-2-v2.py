import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
        
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)

popular_stories.head()

## Question 1
## How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

full_types = hacker_news.query_to_pandas_safe(query)
full_types.head()

## Question 2
## How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """
        
query = """SELECT deleted, COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
            group by deleted
            having count(*) > 1000000
        """

num_comments_deleted = hacker_news.query_to_pandas_safe(query)
num_comments_deleted.head()

## Bonus question
## read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.
# Trying out a hierarchical statement; unfortunately, BigQuery does not support recursion. Here is a messy way to fake it in SQL.
# Probably would be much cleaner to do the recursion in python

query = """with top_level as (
                SELECT id FROM `bigquery-public-data.hacker_news.comments` t1 where not exists (
                    SELECT null FROM `bigquery-public-data.hacker_news.comments` t2 where t1.parent = t2.id)
                ),
                level_1 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join top_level l on t.parent = l.id),
                level_2 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_1 l on t.parent = l.id),
                level_3 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_2 l on t.parent = l.id),
                level_4 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_3 l on t.parent = l.id),
                level_5 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_4 l on t.parent = l.id),
                level_6 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_5 l on t.parent = l.id),
                level_7 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_6 l on t.parent = l.id),
                level_8 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_7 l on t.parent = l.id),
                level_9 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_8 l on t.parent = l.id),
                level_10 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_9 l on t.parent = l.id),
                level_11 as (SELECT t.id FROM `bigquery-public-data.hacker_news.comments` t inner join level_10 l on t.parent = l.id)
            select 0 level, count(*) cnt From top_level
            UNION distinct
            select 1, count(*) from level_1
            UNION distinct
            select 2, count(*) from level_2
            UNION distinct
            select 3, count(*) from level_3
            UNION distinct
            select 4, count(*) from level_4
            UNION distinct
            select 5, count(*) from level_5
            UNION distinct
            select 6, count(*) from level_6
            UNION distinct
            select 7, count(*) from level_7
            UNION distinct
            select 8, count(*) from level_8
            UNION distinct
            select 9, count(*) from level_9
            UNION distinct
            select 10, count(*) from level_10
            UNION distinct
            select 11, count(*) from level_11
            order by 1
        """
        
no_parents = hacker_news.query_to_pandas_safe(query)
no_parents.head(12)

query = """SELECT count(*)
            FROM `bigquery-public-data.hacker_news.comments` a
            where a.id = 92343
        """

comment1 = hacker_news.query_to_pandas_safe(query)
comment1.head()

# query = """with recursive cte (id, parent)
#             SELECT COUNT(*)
#             FROM `bigquery-public-data.hacker_news.comments`
#             where deleted = True
#         """
# 
# num_comments_deleted = hacker_news.query_to_pandas_safe(query)
# num_comments_deleted.head()

## aabi's question
## https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-2?utm_medium=email#282344
query = """SELECT count(*) from (
                select id FROM `bigquery-public-data.hacker_news.full` where type = 'comment'
                except distinct
                select id FROM `bigquery-public-data.hacker_news.comments`
            )
        """
        
query2 = """SELECT count(*) from `bigquery-public-data.hacker_news.comments`
        """        

full_v_comments = hacker_news.query_to_pandas_safe(query)
full_v_comments.head()