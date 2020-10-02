# Getting started with BigQuery on kaggle

# Import BigQuery helper
import bq_helper

# Create object of the BigQuery dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

# Show list of tables
hacker_news.list_tables()

# Check table schema
hacker_news.table_schema("full")

# Check the header of the table
hacker_news.head("full")

# Select rows by columns
hacker_news.head("full", selected_columns="by", num_rows=10)

# Create Query
query="""SELECT score FROM `bigquery-public-data.hacker_news.full` WHERE type="job" """

# Get query size estimation
hacker_news.estimate_query_size(query)

# Cancel the query if query results exceeds limit mentioned in 'max_gb_scanned'
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)

# Run the query
job_post_scores = hacker_news.query_to_pandas_safe(query)

# What is mean of the score (column)?
job_post_scores.score.mean()

# Save it to a csv file (dateset)
job_post_scores.to_csv("job_post_scores.csv")