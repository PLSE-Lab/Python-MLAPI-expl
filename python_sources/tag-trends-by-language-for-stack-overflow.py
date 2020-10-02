#!/usr/bin/env python
# coding: utf-8

# **How to Query the Stack Overflow Data (BigQuery Dataset)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")


# In[ ]:


query = """
WITH posts AS
(
    SELECT
        *
    FROM (
        SELECT
            creation_date
            , SPLIT(TRIM(tags, '<>'), '><') AS tags
        FROM
          `bigquery-public-data.stackoverflow.posts_questions`
        WHERE
          tags <> ''
    ) x
    WHERE EXISTS (SELECT * FROM UNNEST(tags) AS tag WHERE tag = '{0}')
)
SELECT 
    TAG
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2018-01-01') AND PARSE_TIMESTAMP('%F %X','2018-12-31 23:59:59') THEN 1 END) DESC) AS Year2018
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2017-01-01') AND PARSE_TIMESTAMP('%F %X','2017-12-31 23:59:59') THEN 1 END) DESC) AS Year2017
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2016-01-01') AND PARSE_TIMESTAMP('%F %X','2016-12-31 23:59:59') THEN 1 END) DESC) AS Year2016
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2015-01-01') AND PARSE_TIMESTAMP('%F %X','2015-12-31 23:59:59') THEN 1 END) DESC) AS Year2015
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2014-01-01') AND PARSE_TIMESTAMP('%F %X','2014-12-31 23:59:59') THEN 1 END) DESC) AS Year2014
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2013-01-01') AND PARSE_TIMESTAMP('%F %X','2013-12-31 23:59:59') THEN 1 END) DESC) AS Year2013
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2012-01-01') AND PARSE_TIMESTAMP('%F %X','2012-12-31 23:59:59') THEN 1 END) DESC) AS Year2012
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2011-01-01') AND PARSE_TIMESTAMP('%F %X','2011-12-31 23:59:59') THEN 1 END) DESC) AS Year2011
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2010-01-01') AND PARSE_TIMESTAMP('%F %X','2010-12-31 23:59:59') THEN 1 END) DESC) AS Year2010
    , RANK() OVER (ORDER BY COUNT(CASE WHEN creation_date BETWEEN PARSE_TIMESTAMP('%F','2009-01-01') AND PARSE_TIMESTAMP('%F %X','2009-12-31 23:59:59') THEN 1 END) DESC) AS Year2009
FROM 
    posts
    CROSS JOIN UNNEST(posts.tags) as tag
WHERE 
    tag <> '{0}'
GROUP BY
    tag
ORDER BY 
    Year2018
LIMIT 
        30
        """


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("java"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("c#"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("go"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("python"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("php"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("ruby"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("perl"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("javascript"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("rust"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("swift"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("sql"))


# In[ ]:


stackOverflow.query_to_pandas_safe(query.format("typescript"))

