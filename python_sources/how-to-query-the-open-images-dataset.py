#!/usr/bin/env python
# coding: utf-8

# **How to Query the Open Images Dataset (BigQuery)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
open_images = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="open_images")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "open_images")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head("images", num_rows=3)


# In[ ]:


bq_assistant.table_schema("images")


# Which labels are in the dataset?

# In[ ]:


query1 = """SELECT
  *
FROM
  `bigquery-public-data.open_images.dict`
LIMIT
  10;
        """
response1 = open_images.query_to_pandas_safe(query1)
response1.head(10)


# Which labels have "bus" in their display names?

# In[ ]:


query2 = """SELECT
  *
FROM
  `bigquery-public-data.open_images.dict`
WHERE
  label_display_name LIKE '%bus%'
LIMIT
  20;
        """
response2 = open_images.query_to_pandas_safe(query2)
response2.head(10)


# How many images of a trolleybus are in the dataset?

# In[ ]:


query3 = """SELECT
  COUNT(*)
FROM
  `bigquery-public-data.open_images.labels` a
INNER JOIN
  `bigquery-public-data.open_images.images` b
ON
  a.image_id = b.image_id
WHERE
  a.label_name='/m/0f6pl'
  AND a.confidence > 0.5;
        """
response3 = open_images.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)


# What are some landing pages of images with a trolleybus?

# In[ ]:


query4 = """SELECT
  original_landing_url,
  confidence
FROM
  `bigquery-public-data.open_images.labels` l
INNER JOIN
  `bigquery-public-data.open_images.images` i
ON
  l.image_id = i.image_id
WHERE
  label_name='/m/0f6pl'
  AND confidence = 1
  AND subset='validation'
LIMIT
  10;
        """
response4 = open_images.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)


# In[ ]:


query5 = """SELECT
  original_landing_url,
  confidence
FROM
  `bigquery-public-data.open_images.labels` l
INNER JOIN
  `bigquery-public-data.open_images.images` i
ON
  l.image_id = i.image_id
WHERE
  label_name='/m/0f6pl'
  AND confidence = 1
  AND subset='validation'
LIMIT
  10;
        """
response5 = open_images.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(10)


# Which images with cherries are in the training set?

# In[ ]:


query6 = """SELECT
  i.image_id AS image_id,
  original_url,
  confidence
FROM
  `bigquery-public-data.open_images.labels` l
INNER JOIN
  `bigquery-public-data.open_images.images` i
ON
  l.image_id = i.image_id
WHERE
  label_name='/m/0f8sw'
  AND confidence >= 0.85
  AND Subset='train'
LIMIT
  10;
        """
response6 = open_images.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(10)


# Credit: Many functions are adaptations from https://cloud.google.com/bigquery/public-data/openimages

# In[ ]:




