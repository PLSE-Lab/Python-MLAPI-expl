#!/usr/bin/env python
# coding: utf-8

# # About
# 
# This kernel shows how to integrate population and zip code information to the intersections.
# 
# The results are writen to the output **pop_zipcode_intersec.csv** if you want to use it in python right away without BigQuery access. 
# 
# ## Credits
# Some of the ideas are inspired by the following kernels. Please visit them and give them upvotes if you like them.
# - This kernel is a forked from [BigQuery Machine Learning Tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial).

# In[ ]:


# Replace 'kaggle-competitions-project' with YOUR OWN project id here --  
PROJECT_ID = 'kaggle-bq-geotag' #
#PROJECT_ID='kaggle-competitions-project'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_example', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

import seaborn as sns
import matplotlib.pyplot as plt

# create a reference to our table
table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# # Loading population and zip code information

# Let's have a look at the 2010 population of a zip code area

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "SELECT\n    SUM(pop.population) AS population,\n    pop.minimum_age, \n    pop.maximum_age,\n    pop.gender,\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    city,\n    zipcd.zipcode_geom\n  FROM\n    `bigquery-public-data.utility_us.zipcode_area` zipcd,\n    `bigquery-public-data.census_bureau_usa.population_by_zip_2010` pop\n  WHERE\n    zipcd.state_code IN ('MA',\n      'IL',\n      'PA',\n      'GA')\n    AND ( zipcd.city LIKE '%Atlanta%'\n      OR zipcd.city LIKE '%Boston%'\n      OR zipcd.city LIKE '%Chicago%'\n      OR zipcd.city LIKE '%Philadelphia%' )\n    AND SUBSTR(CONCAT('000000', pop.zipcode),-5) = zipcd.zipcode\n  GROUP BY\n    pop.minimum_age, \n    pop.maximum_age,\n    pop.gender,\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    ,\n    zipcd.zipcode_geom\n    limit 100")


# The population is by age and gender. The zip code dataset provides geo information as a polygon.
# 
# Next we check which intersection coordinates are within a polygon. So we can match intersection to zip code.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df', "WITH\n\n  # population per zipcode\n  # (for simplicity ignore gender and age information)\n\n  zip_info AS(\n  SELECT\n    pop.minimum_age, \n    pop.maximum_age,\n    pop.gender,\n    SUM(pop.population) AS population,\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    city,\n    zipcd.zipcode_geom\n  FROM\n    `bigquery-public-data.utility_us.zipcode_area` zipcd,\n    `bigquery-public-data.census_bureau_usa.population_by_zip_2010` pop\n  WHERE\n    zipcd.state_code IN ('MA',\n      'IL',\n      'PA',\n      'GA')\n    AND ( zipcd.city LIKE '%Atlanta%'\n      OR zipcd.city LIKE '%Boston%'\n      OR zipcd.city LIKE '%Chicago%'\n      OR zipcd.city LIKE '%Philadelphia%' )\n    AND SUBSTR(CONCAT('000000', pop.zipcode),-5) = zipcd.zipcode\n  GROUP BY\n    pop.minimum_age, \n    pop.maximum_age,\n    pop.gender,\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    ,\n    zipcd.zipcode_geom),\n  \n  # spatial test and train data\n  \n  train_and_test AS (\n  SELECT\n    tr.intersectionId,\n    tr.longitude,\n    tr.latitude,\n    tr.city\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train` tr\n  UNION DISTINCT\n  SELECT\n    ts.intersectionId,\n    ts.longitude,\n    ts.latitude,\n    ts.city\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.test` ts),\n  \n  # Zipcode and Population per Intersection\n  \n  pop_per_intersection AS (\n  SELECT\n    t.intersectionId,\n    zi.population,\n    zi.zipcode,\n    t.city,\n    zi.minimum_age, \n    zi.maximum_age,\n    zi.gender,\n    zi.zipcode_geom\n  FROM\n    train_and_test t,\n    zip_info zi\n  WHERE\n    t.city = zi.city\n    AND ST_CONTAINS( ST_GEOGFROMTEXT(zi.zipcode_geom),\n      ST_GeogPoint(longitude,\n        latitude)))\n  \n# fill empty/missing zipcodes and population\n\nSELECT\n  t.city,\n  t.intersectionId, \n  p.minimum_age, \n  p.maximum_age,\n  p.gender,\n  coalesce(p.population,\n    round(AVG(p.population) OVER(PARTITION BY t.city, p.minimum_age, p.maximum_age, p.gender))) AS population,\n  coalesce(p.zipcode, 'N/A') AS zipcode,\n  CASE\n    WHEN p.zipcode IS NULL THEN 1\n  ELSE\n  0\nEND AS zip_code_na\n#--,\n#--ST_GeogPoint(t.longitude,\n#--        t.latitude) intersection_gp,\n#p.zipcode_geom\nFROM\n  train_and_test t\nLEFT OUTER JOIN\n  pop_per_intersection p\nON\n  (t.city = p.city\n    AND t.intersectionId = p.intersectionId);")


# In[ ]:


df.head()


# Check number of unique intersections:

# In[ ]:


print('Assert, number of unique intersactions as expected:', df.groupby(['city']).intersectionId.nunique().sum()==6381)


# # Export to csv
# Missing population (where zipcode == 'N/A') is imputed with mean over gender, age and city. Imputed data is flagged in zip_code_na.

# In[ ]:


df.to_csv('pop_zipcode_intersec.csv', index=False)


# In[ ]:




