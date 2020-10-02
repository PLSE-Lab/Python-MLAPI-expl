# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query for finding the number of stories of each type
query1 = """SELECT type, SUM(id) AS total_num
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
type_sum = hacker_news.query_to_pandas_safe(query1)
print(type_sum)

# query for finding the number of comments that were deleted
query2 = """SELECT COUNT(deleted) AS number_of_deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted=True
            """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
print(deleted_comments)