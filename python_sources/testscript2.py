import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_type = hacker_news.query_to_pandas_safe(query2)
stories_type.head()


query3 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = TRUE
        """

deleted_comments = hacker_news.query_to_pandas_safe(query3)
deleted_comments.head()