#!/usr/bin/env python
# coding: utf-8

# **How to Query The Metropolitan Museum of Art's Public Domain Art Works
# (BigQuery Dataset)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
met = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="the_met")


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "the_met")
bq_assistant.list_tables()


# In[ ]:


bq_assistant.head("images", num_rows=15)


# In[ ]:


bq_assistant.table_schema("images")


# What are the types of art by department?
# 

# In[ ]:


query1 = """
SELECT department, COUNT(*) c 
FROM `bigquery-public-data.the_met.objects`
GROUP BY 1
ORDER BY c DESC;
        """
response1 = met.query_to_pandas_safe(query1)
response1.head(10)


# ![](https://cloud.google.com/blog/big-data/2017/08/images/150177792553261/met03.png)
# https://cloud.google.com/blog/big-data/2017/08/images/150177792553261/met03.png

# What are the top types of media used for the items in this collection?

# In[ ]:


query2 = """SELECT 
      LOWER(label) as medium, 
      COUNT(*) c 
FROM `bigquery-public-data.the_met.objects`, 
UNNEST(SPLIT(medium, ',')) label
GROUP BY 1
ORDER BY c DESC;
        """
response2 = met.query_to_pandas_safe(query2)
response2.head(10)


# ![](https://cloud.google.com/blog/big-data/2017/08/images/150177792553261/met04.png)
# https://cloud.google.com/blog/big-data/2017/08/images/150177792553261/met04.png

# What types of art were most popular during various different time periods?

# In[ ]:


query3 = """SELECT period, description, c FROM (
  SELECT 
a.period, 
b.description, 
count(*) c, 
row_number() over (partition by period order by count(*) desc) seqnum 
  FROM `bigquery-public-data.the_met.objects` a
  JOIN (
    SELECT 
        label.description as description, 
        object_id 
    FROM `bigquery-public-data.the_met.vision_api_data`, UNNEST(labelAnnotations) label
  ) b
  ON a.object_id = b.object_id
  WHERE a.period is not null
  group by 1,2
)
WHERE seqnum <= 3
AND c >= 10 # only include labels that have 10 or more pieces associated with it
AND description != "art"
ORDER BY period, c desc;
        """
response3 = met.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(30)


# Where are the images in this dataset currently hosted?

# In[ ]:


query4 = """SELECT REGEXP_EXTRACT(page.url, '//([^/]*)/?') domain, COUNT(*) c
FROM `bigquery-public-data.the_met.vision_api_data`, 
UNNEST(webDetection.pagesWithMatchingImages) as page
GROUP BY 1
ORDER BY c DESC;
        """
response4 = met.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)


# Can you sort images by color?

# In[ ]:


query5 = """SELECT 
color.color.red as r, 
color.color.green as g, 
color.color.blue as b,
concat("https://storage.cloud.google.com/gcs-public-data--met/", 
cast(object_id as string), "/0.jpg") as img_url 
FROM `bigquery-public-data.the_met.vision_api_data`, 
UNNEST(imagePropertiesAnnotation.dominantColors.colors) color
WHERE color.color.red < 0x64
AND (color.color.green > 0x96 or color.color.green < 0xC8)
AND color.color.blue > 0xC8;
        """
response5 = met.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(10)


# Which images contain famous landmarks?
# 

# In[ ]:


query6 = """SELECT b.object_id, b.original_image_url, landmark.description, landmark.mid, landmark.score 
FROM `bigquery-public-data.the_met.vision_api_data` a, 
UNNEST(landmarkAnnotations) landmark
JOIN (
  SELECT object_id, original_image_url, gcs_url 
  FROM `bigquery-public-data.the_met.images` 
) b
ON a.object_id = b.object_id
AND ends_with(lower(b.gcs_url), '/0.jpg')
ORDER BY score DESC;
        """
response6 = met.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(20)


# Credit: Many functions are adaptations of https://cloud.google.com/blog/products/gcp/when-art-meets-big-data-analyzing-200000-items-from-the-met-collection-in-bigquery
