#!/usr/bin/env python
# coding: utf-8

# # INTRO
# 
# This kernel is forked from [BigQuery Machine Learning Tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial).
# 
# Let's use BigQuery to cluster the city intersection based on their location.

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


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'city_df', 'SELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.test` t\nUNION DISTINCT\nSELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` t')


# In[ ]:


cities = list(city_df['city'])
cities


# # Cluster Intersections per City

# Set *model_changed* to True to generate the Models.

# In[ ]:


model_changed = True

if model_changed:
    for c in cities:
        sql="""CREATE MODEL IF NOT EXISTS `bqml_example.model_cluster_"""+c+"""`
        OPTIONS(model_type='kmeans',
                NUM_CLUSTERS = 10) AS
        SELECT
            latitude,
            longitude
        FROM
          `kaggle-competition-datasets.geotab_intersection_congestion.train` t
        WHERE city = '"""+c+"""'
        UNION ALL
        SELECT
            latitude,
            longitude
        FROM
          `kaggle-competition-datasets.geotab_intersection_congestion.test` t
        WHERE city = '"""+c+"""'"""
        
        client.query(sql)

        print('Done with',c)


# # Basic EDA using the clusters

# Using Boston as an example

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'eda_df', 'WITH city_cluster AS (\n    SELECT (SELECT MIN(d.DISTANCE) FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) d) AS dist_to_cluster_center, \n           CONCAT(m.city,"_",CAST(m.CENTROID_ID AS STRING)) AS city_cluster,\n           m.* EXCEPT (nearest_centroids_distance, CENTROID_ID) \n      FROM ML.PREDICT(MODEL `bqml_example.model_cluster_boston`, \n                   (SELECT t.RowId,\n                   t.city,\n                           t.IntersectionId,\n                           t.Latitude,\n                           t.Longitude,\n                           #t.EntryStreetName,\n                           #t.ExitStreetName,\n                           t.EntryHeading,\n                           t.ExitHeading,\n                           t.Hour,\n                           t.Weekend,\n                           t.Month,\n                           #t.Path,\n                           t.TotalTimeStopped_p20,\n                           #t.TotalTimeStopped_p40,\n                           t.TotalTimeStopped_p50,\n                           #t.TotalTimeStopped_p60,\n                           t.TotalTimeStopped_p80,\n                           #t.TimeFromFirstStop_p20,\n                           #t.TimeFromFirstStop_p40,\n                           #t.TimeFromFirstStop_p50,\n                           #t.TimeFromFirstStop_p60,\n                           #t.TimeFromFirstStop_p80,\n                           t.DistanceToFirstStop_p20,\n                           #t.DistanceToFirstStop_p40,\n                           t.DistanceToFirstStop_p50,\n                           #t.DistanceToFirstStop_p60,\n                           t.DistanceToFirstStop_p80,\n                           \'TRAIN\' AS source\n                     FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n                    WHERE city = \'Boston\' \n                    #  AND rowid in(2209678,2209692)\n                    UNION ALL\n                    SELECT t.RowId,\n                    t.city,\n                           t.IntersectionId,\n                           t.Latitude,\n                           t.Longitude,\n                           #t.EntryStreetName,\n                           #t.ExitStreetName,\n                           t.EntryHeading,\n                           t.ExitHeading,\n                           t.Hour,\n                           t.Weekend,\n                           t.Month,\n                           #t.Path,\n                           null as TotalTimeStopped_p20,\n                           #null as TotalTimeStopped_p40,\n                           null as TotalTimeStopped_p50,\n                           #null as TotalTimeStopped_p60,\n                           null as TotalTimeStopped_p80,\n                           #null as TimeFromFirstStop_p20,\n                           #null as TimeFromFirstStop_p40,\n                           #null as TimeFromFirstStop_p50,\n                           #null as TimeFromFirstStop_p60,\n                           #null as TimeFromFirstStop_p80,\n                           null as DistanceToFirstStop_p20,\n                           #null as DistanceToFirstStop_p40,\n                           null as DistanceToFirstStop_p50,\n                           #null as DistanceToFirstStop_p60,\n                           null as DistanceToFirstStop_p80,\n                           \'TEST\' AS source\n                     FROM `kaggle-competition-datasets.geotab_intersection_congestion.test` t\n                    WHERE city = \'Boston\' \n                    #  AND rowid in(2209678,2209692)\n                    )) m\n)\nSELECT cc.source,\n       cc.city,\n       cc.city_cluster, \n       count(1) cnt, \n       avg(cc.dist_to_cluster_center) avg_dist_to_cluster_center, \n       stddev(cc.dist_to_cluster_center) stddev_dist_to_cluster_center, \n       min(cc.dist_to_cluster_center) min_dist_to_cluster_center, \n       max(cc.dist_to_cluster_center) max_dist_to_cluster_center,\n       avg(avg(cc.dist_to_cluster_center)*(count(1)-1)) over(partition by cc.city) avg_dist_to_cluster_center_over_city\n  FROM city_cluster cc\n GROUP BY cc.source, \n          cc.city, \n          cc.city_cluster;')


# In[ ]:


eda_df.sort_values(by=['source','city_cluster']).head(100)


# In[ ]:


sns.swarmplot(x='city_cluster',y='cnt', data=eda_df ,hue='source')

plt.xticks(rotation=45)


# In[ ]:


sns.swarmplot(x='city_cluster',y='avg_dist_to_cluster_center', data=eda_df ,hue='source')

plt.xticks(rotation=45)


# In[ ]:




