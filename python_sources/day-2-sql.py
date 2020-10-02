import bq_helper
hacker = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
dataset_name="hacker_news")
hacker.head("comments")
q1_d2 = """SELECT parent, COUNT(id) FROM `bigquery-public-data.hacker_news.comments` GROUP BY parent HAVING COUNT(id) > 10"""
most_popular = hacker.query_to_pandas_safe(q1_d2)
most_popular.head()
d2_q1 = """SELECT type, COUNT(id) AS Number_of_Stories FROM `bigquery-public-data.hacker_news.full` GROUP BY type"""
type_count=hacker.query_to_pandas_safe(d2_q1)
type_count.head()
d2_q2 = """SELECT COUNT(id) AS Total_deleted FROM `bigquery-public-data.hacker_news.comments` WHERE deleted = True"""
Deleted=hacker.query_to_pandas_safe(d2_q2)
Deleted.head()
d2_op = """SELECT COUNTIF(deleted=True) AS Total_deleted FROM `bigquery-public-data.hacker_news.comments`"""
Optional=hacker.query_to_pandas_safe(d2_op)
Optional.head()