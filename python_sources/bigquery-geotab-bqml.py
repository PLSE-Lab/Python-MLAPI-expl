#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel includes all features and models that lead to my final submission to the [BigQuery-GeoTab Competition](https://www.kaggle.com/c/bigquery-geotab-intersection-congestion).
# 
# My personal objective of the competition is to tryout BigQuery (BQ) including the basic ML features. Therefore the kernel relies as much as possible on BQ. All features are generated in BQ with varying SQL-techniques. The prediction model is also build in BQ. 
# 
# Since the ML features in BQ are currently limited to Lineare Regression, KMeans and Logistic Regression there is a subcompetition within the competition, that ranks top BQML submissions separatly. (There is also a second subcompetition for using TensorFlow with BigQuery.) This kernel aims towards the BQML subcompetition. 
# 
# The kernel uses all the provided ML algorithms:
# - KMeans for Geo-Spatial Clustering
# - Logistic Regression for Adversarial Validation
# - Linear Regression for Prediction Model
# 
# 
# ## Summary
# Features:
# - intersectionid + city (*i_id*)
# - city, hour, weekend, month
# - path, entry_street_name, exit_street_name
# - 20 clusters of intersections per city
# - Distance to the 5 nearest cluster center
# - 8 directions of turn (left, right, centered, uturn, centered-left, ...)
#     - Directions, entry- and exit-heading are not embedded directly. Instead they are translated into degrees (e.g. centered = 90deg). 
#     - Afterwards the degree features are split additionaly into two features sin(feature_x_deg) and cos(feature_x_deg).
#     - Besides the direction of the turn, sin and cos feature are added for entry heading and exit heading 
# - number of intersections per cluster
# - number of observations per city and hour
# - zipcode
# - population of zipcode
# - population of zipcode per intersection
# - intersections per zipcode
# - flag, if entry and exit are the same_street
# - intersection observation frequency
# - approching street length
# - approching street length fallback flag, if approching street length was derived from a more general approach due of missing data
# 
# Features that didn't improve the model:
# - daytime, season (removed from model, since not improving)
# - road_type_entry, road_type_exit (removed from model, since not improving)
# 
# Train-Valid-Split:
# - out of adversarial validation (drop 25 % of not test like data from train) (used by detailed model)
# - split by id range (used by general model)
# 
# Train-Valid-Split that didn't improved the model:
# - remove outliers (target > Q99) from train (removed from model, since not improving)
# 
# Model:
# - Linear Regression
#     - Detailed Model for known path and intersections in train and test
#     - General Model for unknown path and intersections
# 
# ## External data
# The kernel uses zipcodes and population data from the public *bigquery-public-data*-set:
# - bigquery-public-data.utility_us.zipcode_area
# - bigquery-public-data.census_bureau_usa.population_by_zip_2010
# 
# The dataset can be included in the bigquery console (*add ressources*). I shared a preparation of the data (including references to intersections and a csv file for python users) [here](https://www.kaggle.com/joatom/bqml-population-of-zip-code-per-intersection).
# 
# ## Credits
# Some of the ideas are inspired by the following kernels. Please visit them and give them upvotes if you like them.
# - This kernel is a forked from [BigQuery Machine Learning Tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial).
# - The direction features are like the Flow feature in https://www.kaggle.com/jpmiller/intersection-level-eda
# - Intersection encoding in https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb
# - Adversarial Validation: https://www.kaggle.com/tunguz/adversarial-geotab
# - Road encoding: https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others)
# - Same_Street feature: https://www.kaggle.com/ragnar123/feature-engineering-and-forward-feature-selection
# - Remove Path and Intersection for general model: https://www.kaggle.com/gaborfodor/4-xgboost-general
# - Approching Street Lengt inspired by EntryLength: https://www.kaggle.com/dan3dewey/bbq-intersection-congestion

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


## Example to estimate quota usage

# https://stackoverflow.com/questions/47172150/google-cloud-bigquery-python-how-to-get-the-bytes-processed-by-query

sql = """
    SELECT t.city AS city
      FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` t
      LIMIT 20"""

job_config = bigquery.QueryJobConfig()
job_config.dry_run = True

query_job = client.query(sql, job_config)

print ("This query will process: {0:.2f} MB".format(
    query_job.total_bytes_processed/1024/1024))


query_job = client.query(sql)
query_job.result()
print ("Total bytes processed (first execution): {0:.2f} MB".format(
    query_job.total_bytes_processed/1024/1024))


query_job = client.query(sql)
query_job.result()
print ("Total bytes processed (cached): {0:.2f} MB".format(
    query_job.total_bytes_processed/1024/1024))


# In[ ]:


def run_sql(sql, dry=False):
    if dry:
        job_config = bigquery.QueryJobConfig()
        job_config.dry_run = dry
        query_job = client.query(sql, job_config)
        print ("This query will process: {0:.2f} MB".format(query_job.total_bytes_processed/1024/1024))
    else: 
        query_job = client.query(sql)
        query_job.result()
        print ("Total bytes processed (cached): {0:.2f} MB".format(query_job.total_bytes_processed/1024/1024))
        


# # Some basic variables
# To reduce consumption of BQ quota the tables and models are only rebuild if necessary. To rebuild the entire project set **model_changed = True**. Be aware your google account might get charged. 
# 
# If you run on the free sandbox environment you don't have enough quota to run the entire kernel at once. You can then precede the kernel a few days after when your quota has been refreshed. In this case make sure to set the model_changed flag to **False** for the parts you already have built.

# In[ ]:


# for documentation set model_changed here globaly, for development set later in the appropriate cells
model_changed = False


# There will by 12 Models. One for each percentile (20, 50, 80) per category (TotalTimeStopped, DistanceToFirstStop) per model typ (general or detailed).

# In[ ]:


mod_names= ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80',
            'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']

# general model will look like [x+'_general' for x in mod_names]


# Distinct cities

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'city_df --verbose', 'SELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.test` t\nUNION DISTINCT\nSELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` t')


# In[ ]:


cities = list(city_df['city'])
cities


# # Cluster Intersections per City
# Build 20 geo spatial clusters of intersections per city:

# In[ ]:


# model_changed = False

if model_changed:
    for c in cities:
        sql="""DROP MODEL `bqml_example.model_cluster_"""+c+"""`"""
        client.query(sql)

        print('Dropped',c)
    
    
for c in cities:
    sql="""CREATE MODEL IF NOT EXISTS `bqml_example.model_cluster_"""+c+"""`
    OPTIONS(model_type='kmeans',
            NUM_CLUSTERS = 20) AS
    SELECT
        latitude,
        longitude
    FROM
      `kaggle-competition-datasets.geotab_intersection_congestion.train` t
    WHERE city = '"""+c+"""'
    UNION DISTINCT
    SELECT
        latitude,
        longitude
    FROM
      `kaggle-competition-datasets.geotab_intersection_congestion.test` t
    WHERE city = '"""+c+"""'"""

    client.query(sql).result()

    print('Done with',c)


# A basic analysis of the clusters can be found in an earlier kernel [here](https://www.kaggle.com/joatom/bqml-incl-intersection-clusters).

# ## Match closest clusters to intersection
# Now match the five closest cluster center to an intersection. Add distance from intersection to the five cluster center. Count number of intersection of the five surrouding clusters.
# 
# Safe this for further usage in the *CITY_CLUSTER* table.

# In[ ]:


sql=""

for c in cities:
    sql+= """
        SELECT
          city,
          intersectionid,
          CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 1 THEN centroid_id END) AS string)) city_cluster_1,
          CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 2 THEN centroid_id END) AS string)) city_cluster_2,
          CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 3 THEN centroid_id END) AS string)) city_cluster_3,
          CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 4 THEN centroid_id END) AS string)) city_cluster_4,
          CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 5 THEN centroid_id END) AS string)) city_cluster_5,
          MAX(CASE WHEN rk = 1 THEN distance END) dist_to_cluster_center_1,
          MAX(CASE WHEN rk = 2 THEN distance END) dist_to_cluster_center_2,
          MAX(CASE WHEN rk = 3 THEN distance END) dist_to_cluster_center_3,
          MAX(CASE WHEN rk = 4 THEN distance END) dist_to_cluster_center_4,
          MAX(CASE WHEN rk = 5 THEN distance END) dist_to_cluster_center_5,
          COUNT(DISTINCT CONCAT(city, CAST(intersectionid AS string))) OVER(PARTITION BY CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 1 THEN centroid_id END) AS string))) intersections_per_cluster_1,
          COUNT(DISTINCT CONCAT(city, CAST(intersectionid AS string))) OVER(PARTITION BY CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 2 THEN centroid_id END) AS string))) intersections_per_cluster_2,
          COUNT(DISTINCT CONCAT(city, CAST(intersectionid AS string))) OVER(PARTITION BY CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 3 THEN centroid_id END) AS string))) intersections_per_cluster_3,
          COUNT(DISTINCT CONCAT(city, CAST(intersectionid AS string))) OVER(PARTITION BY CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 4 THEN centroid_id END) AS string))) intersections_per_cluster_4,
          COUNT(DISTINCT CONCAT(city, CAST(intersectionid AS string))) OVER(PARTITION BY CONCAT(city,"_", CAST(MAX(CASE WHEN rk = 5 THEN centroid_id END) AS string))) intersections_per_cluster_5
        FROM (
          SELECT
            b.*,
            RANK()OVER(PARTITION BY intersectionid ORDER BY b.distance) rk,
            a.* EXCEPT(centroid_id,
              NEAREST_CENTROIDS_DISTANCE)
          FROM
            ML.PREDICT(MODEL `bqml_example.model_cluster_"""+c+"""`,
              (
              SELECT
                longitude,
                latitude,
                intersectionid,
                city
              FROM
                `kaggle-competition-datasets.geotab_intersection_congestion.train`
              WHERE city = '"""+c+"""'
              UNION DISTINCT
              SELECT
                longitude,
                latitude,
                intersectionid,
                city
              FROM
                `kaggle-competition-datasets.geotab_intersection_congestion.test`
              WHERE city = '"""+c+"""'))a
          CROSS JOIN
            UNNEST(NEAREST_CENTROIDS_DISTANCE)b)
        GROUP BY
          city,
          intersectionid
          
        UNION ALL"""
    
sql = sql[:-len("UNION ALL")]


model_changed = False

if model_changed:
    sql="DROP TABLE IF EXISTS `bqml_example.city_cluster`"
    job_result=client.query(sql).result()

sql = "CREATE TABLE IF NOT EXISTS `bqml_example.city_cluster` as " + sql

job_result=client.query(sql).result()
print('Table CITY_CLUSTER created.')


# ## Train and Test data with one cluster
# The following is a former version of the intersection clustering which combines train and test data with one cluster (the cluster where the intersection is located in). The query results are safed in the tables *CITY_CLUSTER_TRAIN* and *CITY_CLUSTER_TEST* which are used later on for adversarial validation. (Note: The two tables could probably be replaced by *CITY_CLUSTER* after a few changes to the adversarial validation model).

# In[ ]:


def feature_sql(model_name, rowid_split, incl_rowid, tab): 
    
    if incl_rowid:
        rowid = "t.RowId,"
    else:
        rowid = ""
    
    if tab == 'test':
        label = ""
    elif model_name == 'ALL':
        label = """t.TotalTimeStopped_p20,
                t.TotalTimeStopped_p50,
                t.TotalTimeStopped_p80,
                t.DistanceToFirstStop_p20,
                t.DistanceToFirstStop_p50,
                t.DistanceToFirstStop_p80,"""
    else:
        label = """t."""+model_name+""" as label,"""
    
    sql = ""
    
    for c in cities:
        features = """SELECT """+label+"""
                             """+rowid+"""
                             t.city,
                             t.EntryHeading,
                             t.ExitHeading,
                             t.Hour,
                             t.Weekend,
                             t.Month,
                             t.Latitude,
                             t.Longitude,
                             case 
                                 when t.entryheading = t.exitheading THEN
                                  "C"
                                 when ("N" in (t.entryheading, t.exitheading) and "S" in (t.entryheading, t.exitheading)) 
                                      OR 
                                      ("E" in (t.entryheading, t.exitheading) and "W" in (t.entryheading, t.exitheading)) 
                                      OR 
                                      ("NE" in (t.entryheading, t.exitheading) and "SW" in (t.entryheading, t.exitheading))  
                                      OR 
                                      ("SE" in (t.entryheading, t.exitheading) and "NW" in (t.entryheading, t.exitheading)) 
                                 THEN
                                  "U" 
                                 when (t.entryheading="N" and t.exitheading = "W") 
                                      OR(t.entryheading="NW" and t.exitheading = "SW") 
                                      OR(t.entryheading="W" and t.exitheading = "S") 
                                      OR(t.entryheading="SW" and t.exitheading = "SE") 
                                      OR(t.entryheading="S" and t.exitheading = "E") 
                                      OR(t.entryheading="SE" and t.exitheading = "NE") 
                                      OR(t.entryheading="E" and t.exitheading = "N") 
                                      OR(t.entryheading="NE" and t.exitheading = "NW") 
                                 THEN
                                  "L" 
                                 when (t.entryheading="N" and t.exitheading = "E") 
                                      OR(t.entryheading="NW" and t.exitheading = "NE") 
                                      OR(t.entryheading="W" and t.exitheading = "N") 
                                      OR(t.entryheading="SW" and t.exitheading = "NW") 
                                      OR(t.entryheading="S" and t.exitheading = "W") 
                                      OR(t.entryheading="SE" and t.exitheading = "SW") 
                                      OR(t.entryheading="E" and t.exitheading = "S") 
                                      OR(t.entryheading="NE" and t.exitheading = "SE") 
                                 THEN
                                  "R" 
                                 when (t.entryheading="N" and t.exitheading = "NW") 
                                      OR(t.entryheading="NW" and t.exitheading = "W") 
                                      OR(t.entryheading="W" and t.exitheading = "SW") 
                                      OR(t.entryheading="SW" and t.exitheading = "S") 
                                      OR(t.entryheading="S" and t.exitheading = "SE") 
                                      OR(t.entryheading="SE" and t.exitheading = "E") 
                                      OR(t.entryheading="E" and t.exitheading = "NE") 
                                      OR(t.entryheading="NE" and t.exitheading = "N") 
                                 THEN
                                  "CL" 
                                 when (t.entryheading="N" and t.exitheading = "NE") 
                                      OR(t.entryheading="NW" and t.exitheading = "N") 
                                      OR(t.entryheading="W" and t.exitheading = "NW") 
                                      OR(t.entryheading="SW" and t.exitheading = "W") 
                                      OR(t.entryheading="S" and t.exitheading = "SW") 
                                      OR(t.entryheading="SE" and t.exitheading = "S") 
                                      OR(t.entryheading="E" and t.exitheading = "SE") 
                                      OR(t.entryheading="NE" and t.exitheading = "E") 
                                 THEN
                                  "CR" 
                                 when (t.entryheading="N" and t.exitheading = "SW") 
                                      OR(t.entryheading="NW" and t.exitheading = "S") 
                                      OR(t.entryheading="W" and t.exitheading = "SE") 
                                      OR(t.entryheading="SW" and t.exitheading = "E") 
                                      OR(t.entryheading="S" and t.exitheading = "NE") 
                                      OR(t.entryheading="SE" and t.exitheading = "N") 
                                      OR(t.entryheading="E" and t.exitheading = "NW") 
                                      OR(t.entryheading="NE" and t.exitheading = "W") 
                                 THEN
                                  "UL" 
                                 when (t.entryheading="N" and t.exitheading = "SE") 
                                      OR(t.entryheading="NW" and t.exitheading = "E") 
                                      OR(t.entryheading="W" and t.exitheading = "NE") 
                                      OR(t.entryheading="SW" and t.exitheading = "N") 
                                      OR(t.entryheading="S" and t.exitheading = "NW") 
                                      OR(t.entryheading="SE" and t.exitheading = "W") 
                                      OR(t.entryheading="E" and t.exitheading = "SW") 
                                      OR(t.entryheading="NE" and t.exitheading = "S") 
                                 THEN
                                  "UR" 
                               else null end direction
                       FROM `kaggle-competition-datasets.geotab_intersection_congestion."""+tab+"""` t
                      WHERE city = '"""+c+"""' 
                       AND rowid """+rowid_split
                            
        sql += """
               SELECT (SELECT MIN(d.DISTANCE) FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) d) AS dist_to_cluster_center, 
                      CONCAT(m.city,"_",CAST(m.CENTROID_ID AS STRING)) AS city_cluster,
                      m.* EXCEPT (nearest_centroids_distance, CENTROID_ID,Latitude,Longitude) 
                 FROM ML.PREDICT(MODEL `bqml_example.model_cluster_"""+c+"""`, 
                              ("""+features+""")) m
               UNION ALL"""
        
        
    return sql[:-len("UNION ALL")]


# In[ ]:


#model_changed = False

if model_changed:
    sql="DROP TABLE IF EXISTS `bqml_example.city_cluster_train`"
    job_result=client.query(sql).result()
    sql="DROP TABLE IF EXISTS `bqml_example.city_cluster_test`"
    job_result=client.query(sql).result()

    
sql = "CREATE TABLE IF NOT EXISTS `bqml_example.city_cluster_train` as " + feature_sql('ALL','=rowid', True, 'train')
job_result=client.query(sql).result()
sql = "CREATE TABLE IF NOT EXISTS `bqml_example.city_cluster_test` as " + feature_sql('ALL','=rowid', True, 'test')
job_result=client.query(sql).result()
    
print('Done creating CITY_CLUSTER_TRAIN and CITY_CLUSTER_TEST')


# # Complex features

# ## Population of zipcode and observation frequency per intersection
# Retrieve population per zipcode. This kernel does not use age or gender specific information of the underlying data. 
# 
# Extract the features
# 
# - population
# - zipcode,
# - intersections_per_zipcode,
# - pop_intersec_ratio (population / number of intersections of zipcode),
# - zip_code_na (if zipcode is unavailable for intersection) and
# - num_observations (number of intersections in train and test).

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE TABLE IF NOT EXISTS\n  `bqml_example.zipcode_population` AS\nWITH\n\n  # population per zipcode\n\n  zip_info AS(\n  SELECT\n    SUM(pop.population) AS population,\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    city,\n    zipcd.zipcode_geom\n  FROM\n    `bigquery-public-data.utility_us.zipcode_area` zipcd,\n    `bigquery-public-data.census_bureau_usa.population_by_zip_2010` pop\n  WHERE\n    zipcd.state_code IN ('MA',\n      'IL',\n      'PA',\n      'GA')\n    AND ( zipcd.city LIKE '%Atlanta%'\n      OR zipcd.city LIKE '%Boston%'\n      OR zipcd.city LIKE '%Chicago%'\n      OR zipcd.city LIKE '%Philadelphia%' )\n    AND SUBSTR(CONCAT('000000', pop.zipcode),-5) = zipcd.zipcode\n  GROUP BY\n    zipcd.zipcode,\n    CASE zipcd.state_code\n      WHEN 'MA' THEN 'Boston'\n      WHEN 'IL' THEN 'Chicago'\n      WHEN 'GA' THEN 'Atlanta'\n      WHEN 'PA' THEN 'Philadelphia'\n  END\n    ,\n    zipcd.zipcode_geom),\n  \n  # spatial test and train data\n  \n  train_and_test AS (\n  SELECT \n    t_all.intersectionId,\n    t_all.longitude,\n    t_all.latitude,\n    t_all.city,\n    count(1) num_observations\n  FROM (\n      SELECT\n        tr.intersectionId,\n        tr.longitude,\n        tr.latitude,\n        tr.city\n      FROM\n        `kaggle-competition-datasets.geotab_intersection_congestion.train` tr\n      UNION ALL\n      SELECT\n        ts.intersectionId,\n        ts.longitude,\n        ts.latitude,\n        ts.city\n      FROM\n        `kaggle-competition-datasets.geotab_intersection_congestion.test` ts\n      ) t_all\n  GROUP BY\n    t_all.intersectionId,\n    t_all.longitude,\n    t_all.latitude,\n    t_all.city\n  ),\n  \n  # Zipcode and Population per Intersection\n  \n  pop_per_intersection AS (\n  SELECT\n    t.intersectionId,\n    zi.population,\n    zi.zipcode,\n    t.city,\n    COUNT(DISTINCT t.intersectionId) OVER (PARTITION BY zi.zipcode) AS intersections_per_zipcode,\n    round(zi.population / COUNT(DISTINCT t.intersectionId) OVER (PARTITION BY zi.zipcode)) pop_intersec_ratio\n  FROM\n    train_and_test t,\n    zip_info zi\n  WHERE\n    t.city = zi.city\n    AND ST_CONTAINS( ST_GEOGFROMTEXT(zi.zipcode_geom),\n      ST_GeogPoint(longitude,\n        latitude)))\n  \n# fill empty zipcodes and population\n\nSELECT\n  t.city,\n  t.intersectionId,\n  coalesce(p.population,\n    round(AVG(p.population) OVER(PARTITION BY t.city))) AS population,\n  coalesce(p.zipcode, 'N/A') AS zipcode,\n  coalesce(p.intersections_per_zipcode,\n    round(AVG(p.intersections_per_zipcode) OVER(PARTITION BY t.city))) AS intersections_per_zipcode,\n    coalesce(pop_intersec_ratio,round(AVG(p.pop_intersec_ratio) OVER(PARTITION BY t.city))) AS  pop_intersec_ratio,\n  CASE\n    WHEN p.zipcode IS NULL THEN 1\n  ELSE\n    0\n  END AS zip_code_na,\n  t.num_observations\nFROM\n  train_and_test t\nLEFT OUTER JOIN\n  pop_per_intersection p\nON\n  (t.city = p.city\n    AND t.intersectionId = p.intersectionId);")


# ## Length of intersection approaching street
# (The idea of *EntryLength* feature is derived from https://www.kaggle.com/dan3dewey/bbq-intersection-congestion.)
# 
# **Step 1**
# 
# To an entry street/heading find all data sets that contain this entry point as an exit street / heading. For example:
# - Entry: Broadway / NW
# - search all Exit Broadway / NW
# Calculate the distances between matching entry and exit. Make the weak assumption that the distance equals the street length. Pick shortest distance as approaching street length (*app_st_length*).
# 

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', '\ncreate table if not exists `bqml_example.approching_street`  as \n\nWITH\n  # entry and exit data per intersaction from train and test\n  tt AS (\n  SELECT\n    DISTINCT t.IntersectionId,\n    t.Longitude,\n    t.Latitude,\n    t.City,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n  UNION DISTINCT\n  SELECT\n    DISTINCT t.IntersectionId,\n    t.Longitude,\n    t.Latitude,\n    t.City,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.test` t ),\n  # calculate all distances to intersection where exitheading and exitstreet are the same as the current intersections entry\n  # assuming the oncoming street follows the same direction\n  dist_to_t AS(\n  SELECT\n    t.*,\n    st_distance(ST_GEOGPOINT(t.longitude,\n        t.latitude),\n      ST_GEOGPOINT(t_from.longitude,\n        t_from.latitude))dist\n  FROM\n    tt AS t,\n    tt AS t_from\n  WHERE\n    t.city=t_from.city\n    AND t.entrystreetname = t_from.ExitStreetName\n    AND t.EntryHeading = t_from.ExitHeading\n    AND t.Longitude <> t_from.Longitude\n    AND t.Latitude <> t_from.Latitude)\n  # get length of street approching intersection\n  # assuming same direction\nSELECT\n  t.intersectionid,\n  t.city,\n  t.ExitStreetName,\n  t.ExitHeading,\n  t.EntryStreetName,\n  t.EntryHeading,\n  MIN(dist) app_st_length\nFROM\n  dist_to_t t\nGROUP BY\n  t.intersectionid,\n  t.city,\n  t.ExitStreetName,\n  t.ExitHeading,\n  t.EntryStreetName,\n  t.EntryHeading;')


# **Step 2**
# 
# Gather stats to decide how to impute missing *app_st_length*. 
# Examine deviation from existing *app_st_length* to min and average values over several groups (City, City-EntryStreet, City-ExitStreet, City-Intersection).
# Calculate StdDev, Max and Mean for the deviations. Decide for a well balanced fallback solution.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'WITH\n  # get train and test data\n  tt AS(\n  SELECT\n    DISTINCT t.intersectionid,\n    t.city,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n  UNION DISTINCT\n  SELECT\n    DISTINCT t.intersectionid,\n    t.city,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.test` t ),\n  # gather deviation betweewn app_st_length and min/avg values of app_st_length for several partitions\n  devs AS(\n  SELECT\n    #t.*,\n    #round(a.app_st_length, 1) as app_st_length,\n    ABS(ROUND(a.app_st_length, 1)-AVG(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.EntryStreetName ))AS en_avg,\n    ABS(ROUND(a.app_st_length, 1)-MIN(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.EntryStreetName ))AS en_min,\n    ABS(ROUND(a.app_st_length, 1)-AVG(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.IntersectionId ))AS i_avg,\n    ABS(ROUND(a.app_st_length, 1)-MIN(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.IntersectionId ))AS i_min,\n    ABS(ROUND(a.app_st_length, 1)-AVG(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.ExitStreetName ))AS ex_avg,\n    ABS(ROUND(a.app_st_length, 1)- MIN(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city, t.ExitStreetName ))AS ex_min,\n    ABS(ROUND(a.app_st_length, 1)-AVG(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city ))AS c_avg,\n    ABS(ROUND(a.app_st_length, 1)-MIN(ROUND(a.app_st_length, 1) ) OVER (PARTITION BY t.city ))AS c_min\n  FROM\n    tt t\n  LEFT JOIN\n    `bqml_example.approching_street` a\n  ON\n    (t.intersectionid = a.intersectionid\n      AND t.city = a.city\n      AND t.ExitStreetName = a.ExitStreetName\n      AND t.ExitHeading = a.ExitHeading\n      AND t.EntryStreetName = a.EntryStreetName\n      AND t.EntryHeading = a.EntryHeading) )\n  # gather Stddev, Max and Mean for the deviations to decide which order the fallbacks to use\nSELECT\n  stddev(en_avg) stddev_en_avg,\n  stddev(en_min) stddev_en_min,\n  stddev(i_avg) stddev_i_avg,\n  stddev(i_min) stddev_i_min,\n  stddev(ex_avg) stddev_ex_avg,\n  stddev(ex_min) stddev_ex_min,\n  stddev(c_avg) stddev_c_avg,\n  stddev(c_min) stddev_c_min,\n  MIN(en_avg) MIN_en_avg,\n  MIN(en_min) MIN_en_min,\n  MIN(i_avg) MIN_i_avg,\n  MIN(i_min) MIN_i_min,\n  MIN(ex_avg) MIN_ex_avg,\n  MIN(ex_min) MIN_ex_min,\n  MIN(c_avg) MIN_c_avg,\n  MIN(c_min) MIN_c_min,\n  MAX(en_avg) MAX_en_avg,\n  MAX(en_min) MAX_en_min,\n  MAX(i_avg) MAX_i_avg,\n  MAX(i_min) MAX_i_min,\n  MAX(ex_avg) MAX_ex_avg,\n  MAX(ex_min) MAX_ex_min,\n  MAX(c_avg) MAX_c_avg,\n  MAX(c_min) MAX_c_min,\n  AVG(en_avg) AVG_en_avg,\n  AVG(en_min) AVG_en_min,\n  AVG(i_avg) AVG_i_avg,\n  AVG(i_min) AVG_i_min,\n  AVG(ex_avg) AVG_ex_avg,\n  AVG(ex_min) AVG_ex_min,\n  AVG(c_avg) AVG_c_avg,\n  AVG(c_min) AVG_c_min\nFROM\n  devs')


# The fallback order for missing ap_st_length is:
# 1. I_AVG (due to small stddev and avg)
# - EN_AVG
# - EX_AVG
# - C_AVG

# Next, the table *APPROCHING_STREET_IMPUTED* with the approaching street length and the fallback for missing values is created.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'create table if not exists `bqml_example.approching_street_imputed` as\n\n# Get stats to decide which app_st_length fallback to use\nWITH\n  # get train and test data\n  tt AS(\n  SELECT\n    DISTINCT t.intersectionid,\n    t.city,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n  UNION DISTINCT\n  SELECT\n    DISTINCT t.intersectionid,\n    t.city,\n    t.ExitStreetName,\n    t.ExitHeading,\n    t.EntryStreetName,\n    t.EntryHeading\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.test` t )\n  # measures: app_st_length incl. fallbacks\nSELECT\n  t.intersectionid,\n  t.city,\n  t.ExitStreetName,\n  t.ExitHeading,\n  t.EntryStreetName,\n  t.EntryHeading,\n  ROUND(coalesce(a.app_st_length,\n      # fallback I_AVG\n      AVG(a.app_st_length) OVER (PARTITION BY t.city, t.IntersectionId),\n      # fallback EN_AVG\n      AVG(a.app_st_length) OVER (PARTITION BY t.city, t.EntryStreetName),\n      # fallback EX_AVG\n      AVG(a.app_st_length) OVER (PARTITION BY t.city, t.ExitStreetName),\n      # fallback C_AVG\n      AVG(a.app_st_length) OVER (PARTITION BY t.city)), 1) AS app_st_length,\n  CASE\n    WHEN a.app_st_length IS NULL THEN 1\n  ELSE\n  0\nEND\n  app_st_length_fallback\nFROM\n  tt t\nLEFT JOIN\n  `bqml_example.approching_street` a\nON\n  (t.intersectionid = a.intersectionid\n    AND t.city = a.city\n    AND t.ExitStreetName = a.ExitStreetName\n    AND t.ExitHeading = a.ExitHeading\n    AND t.EntryStreetName = a.EntryStreetName\n    AND t.EntryHeading = a.EntryHeading);')


# # Functions for direction features

# This function converts exit and entry headings into turn directions.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE FUNCTION `bqml_example.direction`(entryheading string, exitheading string) \nAS (\n case \n   when entryheading = exitheading THEN\n    "C"\n   when ("N" in (entryheading, exitheading) and "S" in (entryheading, exitheading)) \n        OR \n        ("E" in (entryheading, exitheading) and "W" in (entryheading, exitheading)) \n        OR \n        ("NE" in (entryheading, exitheading) and "SW" in (entryheading, exitheading))  \n        OR \n        ("SE" in (entryheading, exitheading) and "NW" in (entryheading, exitheading)) \n   THEN\n    "U" \n   when (entryheading="N" and exitheading = "W") \n        OR(entryheading="NW" and exitheading = "SW") \n        OR(entryheading="W" and exitheading = "S") \n        OR(entryheading="SW" and exitheading = "SE") \n        OR(entryheading="S" and exitheading = "E") \n        OR(entryheading="SE" and exitheading = "NE") \n        OR(entryheading="E" and exitheading = "N") \n        OR(entryheading="NE" and exitheading = "NW") \n   THEN\n    "L" \n   when (entryheading="N" and exitheading = "E") \n        OR(entryheading="NW" and exitheading = "NE") \n        OR(entryheading="W" and exitheading = "N") \n        OR(entryheading="SW" and exitheading = "NW") \n        OR(entryheading="S" and exitheading = "W") \n        OR(entryheading="SE" and exitheading = "SW") \n        OR(entryheading="E" and exitheading = "S") \n        OR(entryheading="NE" and exitheading = "SE") \n   THEN\n    "R" \n   when (entryheading="N" and exitheading = "NW") \n        OR(entryheading="NW" and exitheading = "W") \n        OR(entryheading="W" and exitheading = "SW") \n        OR(entryheading="SW" and exitheading = "S") \n        OR(entryheading="S" and exitheading = "SE") \n        OR(entryheading="SE" and exitheading = "E") \n        OR(entryheading="E" and exitheading = "NE") \n        OR(entryheading="NE" and exitheading = "N") \n   THEN\n    "CL" \n   when (entryheading="N" and exitheading = "NE") \n        OR(entryheading="NW" and exitheading = "N") \n        OR(entryheading="W" and exitheading = "NW") \n        OR(entryheading="SW" and exitheading = "W") \n        OR(entryheading="S" and exitheading = "SW") \n        OR(entryheading="SE" and exitheading = "S") \n        OR(entryheading="E" and exitheading = "SE") \n        OR(entryheading="NE" and exitheading = "E") \n   THEN\n    "CR" \n   when (entryheading="N" and exitheading = "SW") \n        OR(entryheading="NW" and exitheading = "S") \n        OR(entryheading="W" and exitheading = "SE") \n        OR(entryheading="SW" and exitheading = "E") \n        OR(entryheading="S" and exitheading = "NE") \n        OR(entryheading="SE" and exitheading = "N") \n        OR(entryheading="E" and exitheading = "NW") \n        OR(entryheading="NE" and exitheading = "W") \n   THEN\n    "UL" \n   when (entryheading="N" and exitheading = "SE") \n        OR(entryheading="NW" and exitheading = "E") \n        OR(entryheading="W" and exitheading = "NE") \n        OR(entryheading="SW" and exitheading = "N") \n        OR(entryheading="S" and exitheading = "NW") \n        OR(entryheading="SE" and exitheading = "W") \n        OR(entryheading="E" and exitheading = "SW") \n        OR(entryheading="NE" and exitheading = "S") \n   THEN\n    "UR" \n   else null end\n);')


# This function converts turn directions into degrees, where Center is 90 degrees and uturn 270 degrees.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE FUNCTION `bqml_example.direction2degree`(dir string) AS (\n case dir\n   when "C" then\n    90\n   when \'CL\' then\n    135\n   when "L" then\n    180\n   when \'UL\' then\n    225\n   when "U" then\n    270\n   when \'UR\' then\n    315\n   when "R" then\n    0\n   when \'CR\' then\n    45\n    \n   when "N" then\n    90\n   when \'NW\' then\n    135\n   when "W" then\n    180\n   when \'SW\' then\n    225\n   when "S" then\n    270\n   when \'SE\' then\n    315\n   when "E" then\n    0\n   when \'NE\' then\n    45\n end\n);')


# This function converts road names into road categories. (it's not used in the final model) 

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE FUNCTION `bqml_example.road_encode`(road string) AS (\n case \n    when upper(road) like '%ROAD%' then 'ROAD'\n    when upper(road) like '%STREET%' then 'STREET'\n    when upper(road) like '%AVENUE%' then 'AVENUE'\n    when upper(road) like '%DRIVE%' then 'DRIVE'\n    when upper(road) like '%BROAD%' then 'BROAD'\n    when upper(road) like '%BOULEVARD%' then 'BOULEVARD'\n    else 'OTHERS'\n  end\n);")


# # Test-Validation-Split
# ## Adversarial Validation
# (The approach follows the basic idea of this kernel: https://www.kaggle.com/tunguz/adversarial-geotab)
# 
# **Step 1**
# 
# Try to figure out how similar the test and train data are. For that pick some general features (in the SELECT-part) for test and train and concatenate (UNION ALL) the data. Set the variable *TARGET = 0* for train data and *TARGET = 1* for test. Build a classifier (Logistic Regression model) to predict *TARGET*.

# In[ ]:


# model_changed = False

sql="""
CREATE MODEL IF NOT EXISTS `bqml_example.model_adversarial`
    OPTIONS(MODEL_TYPE='logistic_reg', labels= ['target']) AS
SELECT * FROM
(SELECT 0 as target,
       concat(cc.city,cast(t.intersectionid as string)) i_id,
       cc.city_cluster,
       cc.city,
       cc.hour,
       cc.weekend,
       cc.month,
       cc.direction,
       cc.entryheading,
       cc.exitheading,
       round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,
       round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,
       round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,
       round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,
       round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,
       round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,
       round(cc.dist_to_cluster_center,8) dist_to_cluster_center
 FROM `bqml_example.city_cluster_train` cc,
      `kaggle-competition-datasets.geotab_intersection_congestion.train` t
WHERE t.rowid = cc.rowid
UNION ALL
SELECT 1 as target,
       concat(cc.city,cast(t.intersectionid as string)) i_id,
       cc.city_cluster,
       cc.city,
       cc.hour,
       cc.weekend,
       cc.month,
       cc.direction,
       cc.entryheading,
       cc.exitheading,
       round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,
       round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,
       round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,
       round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,
       round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,
       round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,
       round(cc.dist_to_cluster_center,8) dist_to_cluster_center
 FROM `bqml_example.city_cluster_test` cc,
      `kaggle-competition-datasets.geotab_intersection_congestion.test` t
WHERE t.rowid = cc.rowid)"""

if model_changed:
    run_sql(sql)


# In[ ]:


get_ipython().run_cell_magic('bigquery', '--verbose', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.model_adversarial`)\nORDER BY iteration ')


# **Step 2**
# 
# Evaluate the model to see if it can distinquish between test or train. If not (e.g. accuracy = 0.5) the data is assumed to be similar.

# In[ ]:


#model_changed = False

sql="""SELECT
          *
        FROM ML.EVALUATE(MODEL `bqml_example.model_adversarial`, (
        (SELECT 0 as target,
       concat(cc.city,cast(t.intersectionid as string)) i_id,
       cc.city_cluster,
       cc.city,
       cc.hour,
       cc.weekend,
       cc.month,
       cc.direction,
       cc.entryheading,
       cc.exitheading,
       round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,
       round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,
       round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,
       round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,
       round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,
       round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,
       round(cc.dist_to_cluster_center,8) dist_to_cluster_center
 FROM `bqml_example.city_cluster_train` cc,
      `kaggle-competition-datasets.geotab_intersection_congestion.train` t
WHERE t.rowid = cc.rowid
UNION ALL
SELECT 1 as target,
       concat(cc.city,cast(t.intersectionid as string)) i_id,
       cc.city_cluster,
       cc.city,
       cc.hour,
       cc.weekend,
       cc.month,
       cc.direction,
       cc.entryheading,
       cc.exitheading,
       round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,
       round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,
       round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,
       round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,
       round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,
       round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,
       round(cc.dist_to_cluster_center,8) dist_to_cluster_center
 FROM `bqml_example.city_cluster_test` cc,
      `kaggle-competition-datasets.geotab_intersection_congestion.test` t
WHERE t.rowid = cc.rowid)
        ))"""

if model_changed:
    client.query(sql).to_dataframe()


# Results of adversarial validation:
# 
# * precision: 0.74410540861649255
# * recall: 0.99994428055521567
# * accuracy: 0.7622297087132579
# * f1_score: 0.85326003277995766
# * log_loss: 0.47265372942498035
# * roc_auc: 0.756451
# 
# ==> train and test don't look similar

# ** Step 3 **
# 
# Since train and test data don't look similar we try to get the test-alike train data.
# 
# Create table with adversarial probability of train beeing classified as test:

# In[ ]:


model_changed = False

sql="""
create table if not exists `bqml_example.testalike_av` as
SELECT
          rowid,
          (SELECT prob FROM UNNEST(predicted_target_probs) WHERE label=0 LIMIT 1) as prob_train,
          (SELECT prob FROM UNNEST(predicted_target_probs) WHERE label=1 LIMIT 1) as prob_test
        FROM ML.PREDICT(MODEL `bqml_example.model_adversarial`, (
        (
SELECT concat(cc.city,cast(t.intersectionid as string)) i_id,
       cc.city_cluster,
       cc.city,
       cc.hour,
       cc.weekend,
       cc.month,
       cc.direction,
       cc.entryheading,
       cc.exitheading,
       round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,
       round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,
       round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,
       round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,
       round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,
       round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,
       round(cc.dist_to_cluster_center,8) dist_to_cluster_center,
       t.rowid
 FROM `bqml_example.city_cluster_train` cc,
      `kaggle-competition-datasets.geotab_intersection_congestion.train` t
WHERE t.rowid = cc.rowid)
        ))"""

if model_changed:
    run_sql(sql) #,dry=True)


# Examin the test-alike probability of the train data:

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df_av --verbose', 'select cc.city, cc.city_cluster, av.prob_test \n  from `bqml_example.testalike_av` av,\n       `bqml_example.city_cluster_train` cc\n where cc.rowid = av.rowid')


# In[ ]:


df_av.hist()


# In[ ]:


df_av.hist(by='city')


# In[ ]:


df_av.describe()


# ** Step 4 **
# 
# There are about 200000 train data entries with a 0.1 probability to be similar to test. Looking at the histograms we dismiss every data entry with less then 0.5 test-probability for the detailed model later on.
# The 0.5 threshold out of adversarial validation only worked on the detailed model. The general model did a simple validation-split by row number.

# ## Remove Outliers from train
# **!! This approach unfortunately didn't advance the model and was left out for the final model !!**
# 
# The main idea is to analyse the target values (TotalTimeStopped and DistanceToFirstStop) and considere values above the 99.x-quantil as outliers and use this value as either a cutoff (clip)-value for the predicted targets or remove them from the train data. In both cases the model didn't improve.
# 
# ** Step 1 **
# 
# Mark rows with target value > 99.5-percentile. Group by City-Cluster-1 to calculate quantil.
# 
# (I needed to split SQL because of CPU-usage restriction on free tier.)

# ### TotalTimeStopped
# using q=0.995

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'tts_q', 'SELECT\nc.city_cluster_1,\ncount(1)cnt,\nmin(TotalTimeStopped_p20) tts20_min,\nmax(TotalTimeStopped_p20) tts20_max,\navg(TotalTimeStopped_p20) tts20_avg,\nstddev(TotalTimeStopped_p20) tts20_std,\nAPPROX_QUANTILES(TotalTimeStopped_p20,201)[SAFE_ORDINAL(200)] tts20_q99,\n\nmin(TotalTimeStopped_p50) tts50_min,\nmax(TotalTimeStopped_p50) tts50_max,\navg(TotalTimeStopped_p50) tts50_avg,\nstddev(TotalTimeStopped_p50) tts50_std,\nAPPROX_QUANTILES(TotalTimeStopped_p50,201)[SAFE_ORDINAL(200)] tts50_q99,\n\nmin(TotalTimeStopped_p80) tts80_min,\nmax(TotalTimeStopped_p80) tts80_max,\navg(TotalTimeStopped_p80) tts80_avg,\nstddev(TotalTimeStopped_p80) tts80_std,\nAPPROX_QUANTILES(TotalTimeStopped_p80,201)[SAFE_ORDINAL(200)] tts80_q99\n\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n  `bqml_example.city_cluster` c\nWHERE\n  c.city = c.city\n  AND t.intersectionid=t.intersectionid\nGROUP BY\n  c.city_cluster_1;')


# In[ ]:


tts_q.describe()


# Since stddev of q99 is small I use the same filter values for all city clusters.
# 
# Define filter to remove outliers:

# In[ ]:


tts20_outlier_filter = tts_q['tts20_q99'].max()
tts50_outlier_filter = tts_q['tts50_q99'].max()
tts80_outlier_filter = tts_q['tts80_q99'].max()

print('TotalTimeStopped_20 filter:', tts20_outlier_filter)
print('TotalTimeStopped_50 filter:', tts50_outlier_filter)
print('TotalTimeStopped_80 filter:', tts80_outlier_filter)


# ### DistanceToFirstStop
# q=0.995

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'dts_q20', 'SELECT\nc.city_cluster_1,\ncount(1)cnt,\nmin(DistanceToFirstStop_p20) dts20_min,\nmax(DistanceToFirstStop_p20) dts20_max,\navg(DistanceToFirstStop_p20) dts20_avg,\nstddev(DistanceToFirstStop_p20) dts20_std,\nAPPROX_QUANTILES(DistanceToFirstStop_p20,201)[SAFE_ORDINAL(200)] dts20_q99\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n  `bqml_example.city_cluster` c\nWHERE\n  c.city = c.city\n  AND t.intersectionid=t.intersectionid\nGROUP BY\n  c.city_cluster_1')


# In[ ]:


dts_q20.describe()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'dts_q50', 'SELECT\nc.city_cluster_1,\ncount(1)cnt,\nmin(DistanceToFirstStop_p50) dts50_min,\nmax(DistanceToFirstStop_p50) dts50_max,\navg(DistanceToFirstStop_p50) dts50_avg,\nstddev(DistanceToFirstStop_p50) dts50_std,\nAPPROX_QUANTILES(DistanceToFirstStop_p50,201)[SAFE_ORDINAL(200)] dts50_q99\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n  `bqml_example.city_cluster` c\nWHERE\n  c.city = c.city\n  AND t.intersectionid=t.intersectionid\nGROUP BY\n  c.city_cluster_1;')


# In[ ]:


dts_q50.describe()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'dts_q80', 'SELECT\nc.city_cluster_1,\ncount(1)cnt,\nmin(DistanceToFirstStop_p80) dts80_min,\nmax(DistanceToFirstStop_p80) dts80_max,\navg(DistanceToFirstStop_p80) dts80_avg,\nstddev(DistanceToFirstStop_p80) dts80_std,\nAPPROX_QUANTILES(DistanceToFirstStop_p80,201)[SAFE_ORDINAL(200)] dts80_q99\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n  `bqml_example.city_cluster` c\nWHERE\n  c.city = c.city\n  AND t.intersectionid=t.intersectionid\nGROUP BY\n  c.city_cluster_1;')


# In[ ]:


dts_q80.describe()


# Since stddev of q99 is small I use the same filter values for all city clusters.
# 
# Define filter to remove outliers:

# In[ ]:


dts20_outlier_filter = dts_q20['dts20_q99'].max()
dts50_outlier_filter = dts_q50['dts50_q99'].max()
dts80_outlier_filter = dts_q80['dts80_q99'].max()

print('DistanceToFirstStop_20 filter:', dts20_outlier_filter)
print('DistanceToFirstStop_50 filter:', dts50_outlier_filter)
print('DistanceToFirstStop_80 filter:', dts80_outlier_filter)


# In[ ]:


outliers_filter = {'TotalTimeStopped_p20':tts20_outlier_filter,
                   'TotalTimeStopped_p50':tts50_outlier_filter,
                   'TotalTimeStopped_p80':tts80_outlier_filter,
                   'DistanceToFirstStop_p20':dts20_outlier_filter,
                   'DistanceToFirstStop_p50':dts50_outlier_filter,
                   'DistanceToFirstStop_p80':dts80_outlier_filter}

outliers_filter


# # Build models
# (The idea of creating a general and a detailed model depending on intersection and street informations was derived from https://www.kaggle.com/gaborfodor/4-xgboost-general)

# ## Detailed model
# The detailed prediction model is trained with data where the intersections and streets are also known in the test data set.
# The entire dataset above the 0.5 adversarial threshold is used for train. Usualy I'd keep some amount for validation. But I dismissed it (as well as cross validatin) because of limited BQ quota. I only tried a couple of different hyper parameters for the same reason. The default settings fit best.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#model_changed = False\n\nif model_changed:\n    print("Let\'s go")\n\n    for mn in mod_names:\n        sql="""\n        CREATE OR REPLACE MODEL `bqml_example.model_"""+mn+"""`\n        OPTIONS(MODEL_TYPE=\'linear_reg\') AS \n        SELECT  t."""+mn+""" as label,\n                concat(t.city,cast(t.intersectionid as string)) i_id,\n                cc.city_cluster_1,\n                cc.city_cluster_2,\n                cc.city_cluster_3,\n                cc.city_cluster_4,\n                cc.city_cluster_5,\n                t.city,\n                t.hour,\n                t.weekend,\n                t.month,\n                --(case \n                --    when t.hour between 6 and 9 then \'RUSH_HOUR_MORNING\'\n                --    when t.hour between 10 and 15 then \'MIDDAY\'\n                --    when t.hour between 16 and 19 then \'RUSH_HOUR_EVENING\'\n                --    else \'NIGHT\'\n                --end) daytime,\n                --(case \n                --    when t.month between 4 and 6 then \'SPRING\'\n                --    when t.month between 7 and 9 then \'SUMMER\'\n                --    when t.month between 10 and 11 then \'FALL\'\n                --    else \'WINTER\'\n                --end) season,\n                `bqml_example.direction`(t.entryheading, t.exitheading) direction,\n                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,\n                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,\n                t.entryheading,\n                t.exitheading,\n                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,\n                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,\n                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,\n                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,\n                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,\n                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,\n                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,\n                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,\n                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,\n                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,\n                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,\n                intersections_per_cluster_1,\n                intersections_per_cluster_2,\n                intersections_per_cluster_3,\n                intersections_per_cluster_4,\n                intersections_per_cluster_5,\n                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,\n                zp.population,\n                zp.zipcode,\n                zp.intersections_per_zipcode,\n                zp.pop_intersec_ratio,\n                zp.zip_code_na,\n                concat(t.city,t.path) as path,\n                concat(t.city,t.entryStreetName) as entry_street_name,\n                concat(t.city,t.exitStreetName) as exit_street_name,\n                case when t.entryStreetName = t.exitStreetName then\n                    1\n                else\n                    0\n                end as same_street,\n                zp.num_observations,\n                a.app_st_length,\n                a.app_st_length_fallback\n          FROM `bqml_example.city_cluster` cc,\n               `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n               `bqml_example.testalike_av` av,\n               `bqml_example.zipcode_population` zp,\n               `bqml_example.approching_street_imputed` a\n         WHERE t.city = cc.city\n           AND t.intersectionid = cc.intersectionid\n           AND t.intersectionid = zp.intersectionid\n           AND t.city = zp.city\n           AND t.rowid = av.rowid\n           AND av.prob_test > 0.5\n           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)\n           AND t.city = a.city\n           AND ifnull(t.ExitStreetName,\'#_#\') = ifnull(a.ExitStreetName,\'#_#\')\n           AND ifnull(t.ExitHeading,\'#_#\') = ifnull(a.ExitHeading,\'#_#\')\n           AND ifnull(t.EntryStreetName,\'#_#\') = ifnull(a.EntryStreetName,\'#_#\')\n           AND ifnull(t.EntryHeading,\'#_#\') = ifnull(a.EntryHeading,\'#_#\')\n        """\n        \n        client.query(sql).result()\n\n        print(\'Done with\',mn)')


# ## General Model
# 
# For the test data that doesn't match intersections, path, entry and exit streets from train I build a more general model. BATCH_GRADIENT_DESCENT needs to be forced on this model, otherwise with the default optimize strategie the outliers in the prediction are very high. For this model the train-validation-split is done by rowid (split by adversiarial validation didn't improve the general model).

# In[ ]:


get_ipython().run_cell_magic('time', '', '#model_changed = False\n\nif model_changed:\n    print("Let\'s go")\n\n    for mn in mod_names:\n        sql="""\n        CREATE OR REPLACE MODEL `bqml_example.model_"""+mn+"""_general`\n        OPTIONS(MODEL_TYPE=\'linear_reg\',L2_REG=0.2,\n                LS_INIT_LEARN_RATE=0.4,\n                OPTIMIZE_STRATEGY=\'BATCH_GRADIENT_DESCENT\') AS \n        SELECT  t."""+mn+""" as label,\n                t.Longitude,\n                t.Latitude, \n                cc.city_cluster_1,\n                cc.city_cluster_2,\n                cc.city_cluster_3,\n                cc.city_cluster_4,\n                cc.city_cluster_5,\n                t.city,\n                t.hour,\n                t.weekend,\n                t.month,\n                `bqml_example.direction`(t.entryheading, t.exitheading) direction,\n                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,\n                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,\n                t.entryheading,\n                t.exitheading,\n                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,\n                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,\n                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,\n                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,\n                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,\n                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,\n                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,\n                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,\n                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,\n                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,\n                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,\n                intersections_per_cluster_1,\n                intersections_per_cluster_2,\n                intersections_per_cluster_3,\n                intersections_per_cluster_4,\n                intersections_per_cluster_5,\n                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,\n                zp.population,\n                zp.zipcode,\n                zp.intersections_per_zipcode,\n                zp.pop_intersec_ratio,\n                zp.zip_code_na,\n                case when t.entryStreetName = t.exitStreetName then\n                    1\n                else\n                    0\n                end as same_street,\n                zp.num_observations,\n                a.app_st_length,\n                a.app_st_length_fallback\n          FROM `bqml_example.city_cluster` cc,\n               `kaggle-competition-datasets.geotab_intersection_congestion.train` t,\n               `bqml_example.testalike_av` av,\n               `bqml_example.zipcode_population` zp,\n               `bqml_example.approching_street_imputed` a\n         WHERE t.city = cc.city\n           AND t.intersectionid = cc.intersectionid\n           AND t.intersectionid = zp.intersectionid\n           AND t.city = zp.city\n           AND t.rowid = av.rowid\n           and t.rowid < 2600000\n           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)\n           AND t.city = a.city\n           AND ifnull(t.ExitStreetName,\'#_#\') = ifnull(a.ExitStreetName,\'#_#\')\n           AND ifnull(t.ExitHeading,\'#_#\') = ifnull(a.ExitHeading,\'#_#\')\n           AND ifnull(t.EntryStreetName,\'#_#\') = ifnull(a.EntryStreetName,\'#_#\')\n           AND ifnull(t.EntryHeading,\'#_#\') = ifnull(a.EntryHeading,\'#_#\')\n        """\n        #\n         #  AND av.prob_test > 0.5\n         #  AND t."""+mn+""" <= """+str(outliers_filter[mn])+"""\n        #between 0.5 and 0.85\n    #< 2600000;\n        client.query(sql).result()\n\n        print(\'Done with\',mn)')


# ## Get training statistics
# For example TotalTimeStopped_p20 is shown in the notebook. Evaluation on the other models was done for convenience in the BigQuery console.

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.model_TotalTimeStopped_p20`)\nORDER BY iteration ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.model_TotalTimeStopped_p20_general`)\nORDER BY iteration ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.FEATURE_INFO(MODEL `bqml_example.model_TotalTimeStopped_p20`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.WEIGHTS(MODEL  `bqml_example.model_TotalTimeStopped_p20`,\n    STRUCT(true AS standardize))')


# ## Evaluating the model
# Since I didn't leave out an appropriate validation set (e.g. top 10% of the test-alike train data) the evaluation is not a good indicator for the leaderboard. I rather used it briefly to check for slight improvments to the previous run. This is definitely a shortcoming of this kernel.
# 

# ### General model

# In[ ]:


# Using data below adversarial threshold as validation set :-/ 
sql="""SELECT
          *
        FROM ML.EVALUATE(MODEL `bqml_example.model_TotalTimeStopped_p20`, (
        SELECT  t.TotalTimeStopped_p20 as label, 
                concat(t.city,cast(t.intersectionid as string)) i_id,
                cc.city_cluster_1,
                cc.city_cluster_2,
                cc.city_cluster_3,
                cc.city_cluster_4,
                cc.city_cluster_5,
                t.city,
                t.hour,
                t.weekend,
                t.month,
                --(case 
                --    when t.hour between 6 and 9 then 'RUSH_HOUR_MORNING'
                --   when t.hour between 10 and 15 then 'MIDDAY'
                --    when t.hour between 16 and 19 then 'RUSH_HOUR_EVENING'
                --    else 'NIGHT'
                --end) daytime,
                --(case 
                --    when t.month between 4 and 6 then 'SPRING'
                --    when t.month between 7 and 9 then 'SUMMER'
                --    when t.month between 10 and 11 then 'FALL'
                --    else 'WINTER'
                --end) season,
                `bqml_example.direction`(t.entryheading, t.exitheading) direction,
                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,
                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,
                t.entryheading,
                t.exitheading,
                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,
                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,
                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,
                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,
                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,
                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,
                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,
                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,
                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,
                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,
                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,
                intersections_per_cluster_1,
                intersections_per_cluster_2,
                intersections_per_cluster_3,
                intersections_per_cluster_4,
                intersections_per_cluster_5,
                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,
                zp.population,
                zp.zipcode,
                zp.intersections_per_zipcode,
                zp.pop_intersec_ratio,
                zp.zip_code_na,
                concat(t.city,t.path) as path,
                concat(t.city,t.entryStreetName) as entry_street_name,
                concat(t.city,t.exitStreetName) as exit_street_name,
                case when t.entryStreetName = t.exitStreetName then
                    1
                else
                    0
                end as same_street,
                zp.num_observations,
                a.app_st_length,
                a.app_st_length_fallback
  FROM `bqml_example.city_cluster` cc,
               `kaggle-competition-datasets.geotab_intersection_congestion.train` t,
               `bqml_example.testalike_av` av,
               `bqml_example.zipcode_population` zp,
               `bqml_example.approching_street_imputed` a
         WHERE t.city = cc.city
           AND t.intersectionid = cc.intersectionid
           AND t.intersectionid = zp.intersectionid
           AND t.city = zp.city
           AND t.rowid = av.rowid
           AND av.prob_test <= 0.5
           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)
           AND t.city = a.city
           AND ifnull(t.ExitStreetName,'#_#') = ifnull(a.ExitStreetName,'#_#')
           AND ifnull(t.ExitHeading,'#_#') = ifnull(a.ExitHeading,'#_#')
           AND ifnull(t.EntryStreetName,'#_#') = ifnull(a.EntryStreetName,'#_#')
           AND ifnull(t.EntryHeading,'#_#') = ifnull(a.EntryHeading,'#_#')))"""

client.query(sql).to_dataframe()


# ### General model

# In[ ]:


# validation set by rowid
sql="""SELECT
          *
        FROM ML.EVALUATE(MODEL `bqml_example.model_TotalTimeStopped_p20_general`, (
        SELECT  t.TotalTimeStopped_p20 as label, 
                t.Longitude,
                t.Latitude, 
                cc.city_cluster_1,
                cc.city_cluster_2,
                cc.city_cluster_3,
                cc.city_cluster_4,
                cc.city_cluster_5,
                t.city,
                t.hour,
                t.weekend,
                t.month,
                `bqml_example.direction`(t.entryheading, t.exitheading) direction,
                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,
                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,
                t.entryheading,
                t.exitheading,
                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,
                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,
                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,
                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,
                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,
                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,
                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,
                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,
                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,
                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,
                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,
                intersections_per_cluster_1,
                intersections_per_cluster_2,
                intersections_per_cluster_3,
                intersections_per_cluster_4,
                intersections_per_cluster_5,
                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,
                zp.population,
                zp.zipcode,
                zp.intersections_per_zipcode,
                zp.pop_intersec_ratio,
                zp.zip_code_na,
                case when t.entryStreetName = t.exitStreetName then
                    1
                else
                    0
                end as same_street,
                zp.num_observations,
                a.app_st_length,
                a.app_st_length_fallback
  FROM `bqml_example.city_cluster` cc,
               `kaggle-competition-datasets.geotab_intersection_congestion.train` t,
               `bqml_example.testalike_av` av,
               `bqml_example.zipcode_population` zp,
               `bqml_example.approching_street_imputed` a
         WHERE t.city = cc.city
           AND t.intersectionid = cc.intersectionid
           AND t.intersectionid = zp.intersectionid
           AND t.city = zp.city
           AND t.rowid = av.rowid
           and t.rowid >= 2600000
           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)
           AND t.city = a.city
           AND ifnull(t.ExitStreetName,'#_#') = ifnull(a.ExitStreetName,'#_#')
           AND ifnull(t.ExitHeading,'#_#') = ifnull(a.ExitHeading,'#_#')
           AND ifnull(t.EntryStreetName,'#_#') = ifnull(a.EntryStreetName,'#_#')
           AND ifnull(t.EntryHeading,'#_#') = ifnull(a.EntryHeading,'#_#')
           ))"""

#AND av.prob_test <= 0.5

client.query(sql).to_dataframe()


# # Predict outcomes
# Predict the target values and generate the output file. Use detailed model for test data with intersection_id, path, exit and entry street existing in train. Use general model otherwise. Clip negative values to 0.
# 

# In[ ]:


def pred(mn, debug=False):
    
    if debug:
        lmt='LIMIT 10'
    else:
        lmt=''
    
    ## Detailed model
    
    sql="""
    SELECT
      RowId,
      case when predicted_label < 0 then 
          0 
      else 
          predicted_label 
      end as """+mn+"""
    FROM
      ML.PREDICT(MODEL `bqml_example.model_"""+mn+"""`,
        (
        SELECT  t.RowId, 
                concat(t.city,cast(t.intersectionid as string)) i_id,
                cc.city_cluster_1,
                cc.city_cluster_2,
                cc.city_cluster_3,
                cc.city_cluster_4,
                cc.city_cluster_5,
                t.city,
                t.hour,
                t.weekend,
                t.month,
                --(case 
                --    when t.hour between 6 and 9 then 'RUSH_HOUR_MORNING'
                --    when t.hour between 10 and 15 then 'MIDDAY'
                --    when t.hour between 16 and 19 then 'RUSH_HOUR_EVENING'
                --    else 'NIGHT'
                --end) daytime,
                --(case 
                --    when t.month between 4 and 6 then 'SPRING'
                --    when t.month between 7 and 9 then 'SUMMER'
                --    when t.month between 10 and 11 then 'FALL'
                --    else 'WINTER'
                --end) season,
                `bqml_example.direction`(t.entryheading, t.exitheading) direction,
                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,
                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,
                t.entryheading,
                t.exitheading,
                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,
                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,
                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,
                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,
                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,
                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,
                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,
                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,
                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,
                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,
                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,
                intersections_per_cluster_1,
                intersections_per_cluster_2,
                intersections_per_cluster_3,
                intersections_per_cluster_4,
                intersections_per_cluster_5,
                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,
                zp.population,
                zp.zipcode,
                zp.intersections_per_zipcode,
                zp.pop_intersec_ratio,
                zp.zip_code_na,
                concat(t.city,t.path) as path,
                concat(t.city,t.entryStreetName) as entry_street_name,
                concat(t.city,t.exitStreetName) as exit_street_name,
                case when t.entryStreetName = t.exitStreetName then
                    1
                else
                    0
                end as same_street,
                zp.num_observations,
                a.app_st_length,
                a.app_st_length_fallback
          FROM `bqml_example.city_cluster` cc,
               `kaggle-competition-datasets.geotab_intersection_congestion.test` t,
               `bqml_example.zipcode_population` zp,
               `bqml_example.approching_street_imputed` a
         WHERE t.city = cc.city
           AND t.intersectionid = cc.intersectionid
           AND t.intersectionid = zp.intersectionid
           AND t.city = zp.city
           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)
           AND t.city = a.city
           AND ifnull(t.ExitStreetName,'#_#') = ifnull(a.ExitStreetName,'#_#')
           AND ifnull(t.ExitHeading,'#_#') = ifnull(a.ExitHeading,'#_#')
           AND ifnull(t.EntryStreetName,'#_#') = ifnull(a.EntryStreetName,'#_#')
           AND ifnull(t.EntryHeading,'#_#') = ifnull(a.EntryHeading,'#_#')
           AND    EXISTS (
                  SELECT
                    1
                  FROM
                    `kaggle-competition-datasets.geotab_intersection_congestion.train` n
                  WHERE
                    n.IntersectionId = t.IntersectionId
                    and n.EntryStreetName = t.EntryStreetName
                    AND n.ExitStreetName = t.ExitStreetName
                    AND n.Path = t.path)
                          
          """+lmt+"""))
        ORDER BY RowId ASC"""
    
    df=client.query(sql).to_dataframe()

    ## general model
    
    sql="""
    SELECT
      RowId,
      case when predicted_label < 0 then 
          0 
      else 
          predicted_label 
      end as """+mn+"""
    FROM
      ML.PREDICT(MODEL `bqml_example.model_"""+mn+"""_general`,
        (
        SELECT  t.RowId, 
                t.Longitude,
                t.Latitude, 
                cc.city_cluster_1,
                cc.city_cluster_2,
                cc.city_cluster_3,
                cc.city_cluster_4,
                cc.city_cluster_5,
                t.city,
                t.hour,
                t.weekend,
                t.month,
                `bqml_example.direction`(t.entryheading, t.exitheading) direction,
                --`bqml_example.road_encode`(t.entrystreetname) road_type_entry,
                --`bqml_example.road_encode`(t.exitstreetname) road_type_exit,
                t.entryheading,
                t.exitheading,
                round(sin(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_sin,
                round(cos(bqml_example.direction2degree(`bqml_example.direction`(t.entryheading, t.exitheading))*ACOS(-1)/180),6) direction_cos,
                round(sin(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_sin,
                round(cos(bqml_example.direction2degree(t.entryheading)*ACOS(-1)/180),6) entryheading_cos,
                round(sin(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_sin,
                round(cos(bqml_example.direction2degree(t.exitheading)*ACOS(-1)/180),6) exitheading_cos,
                round(cc.dist_to_cluster_center_1,8) dist_to_cluster_center_1,
                round(cc.dist_to_cluster_center_2,8) dist_to_cluster_center_2,
                round(cc.dist_to_cluster_center_3,8) dist_to_cluster_center_3,
                round(cc.dist_to_cluster_center_4,8) dist_to_cluster_center_4,
                round(cc.dist_to_cluster_center_5,8) dist_to_cluster_center_5,
                intersections_per_cluster_1,
                intersections_per_cluster_2,
                intersections_per_cluster_3,
                intersections_per_cluster_4,
                intersections_per_cluster_5,
                count(1)over(partition by t.city, t.hour) / count(1)over(partition by t.city) observation_ratio_per_city,
                zp.population,
                zp.zipcode,
                zp.intersections_per_zipcode,
                zp.pop_intersec_ratio,
                zp.zip_code_na,
                case when t.entryStreetName = t.exitStreetName then
                    1
                else
                    0
                end as same_street,
                zp.num_observations,
                a.app_st_length,
                a.app_st_length_fallback
          FROM `bqml_example.city_cluster` cc,
               `kaggle-competition-datasets.geotab_intersection_congestion.test` t,
               `bqml_example.zipcode_population` zp,
               `bqml_example.approching_street_imputed` a
         WHERE t.city = cc.city
           AND t.intersectionid = cc.intersectionid
           AND t.intersectionid = zp.intersectionid
           AND t.city = zp.city
           AND ifnull(t.intersectionid, -99 ) = ifnull(a.intersectionid,-99)
           AND t.city = a.city
           AND ifnull(t.ExitStreetName,'#_#') = ifnull(a.ExitStreetName,'#_#')
           AND ifnull(t.ExitHeading,'#_#') = ifnull(a.ExitHeading,'#_#')
           AND ifnull(t.EntryStreetName,'#_#') = ifnull(a.EntryStreetName,'#_#')
           AND ifnull(t.EntryHeading,'#_#') = ifnull(a.EntryHeading,'#_#')
           AND    NOT EXISTS (
                  SELECT
                    1
                  FROM
                    `kaggle-competition-datasets.geotab_intersection_congestion.train` n
                  WHERE
                    n.IntersectionId = t.IntersectionId
                    and n.EntryStreetName = t.EntryStreetName
                    AND n.ExitStreetName = t.ExitStreetName
                    AND n.Path = t.path)
                            
          """+lmt+"""))
        ORDER BY RowId ASC"""
        
    df=df.append(client.query(sql).to_dataframe())
        
    return df
    
df=None
for i, mn in enumerate(mod_names):
    if i == 0:
        print('Start', i)
        df = pred(mn)
        df['RowId'] = df['RowId'].apply(str) + '_'+str(i)
        df.rename(columns={'RowId': 'TargetId', mn: 'Target'}, inplace=True)
    else:
        print('Start', i)
        df_temp = pred(mn)
        df_temp['RowId'] = df_temp['RowId'].apply(str) + '_'+str(i)
        df_temp.rename(columns={'RowId': 'TargetId', mn: 'Target'}, inplace=True)
        df=df.append(df_temp)

    print('Done with',mn)


# In[ ]:


print(df.shape)
df.head(100)


# ## Output as CSV
# 

# In[ ]:


df.to_csv('submission.csv', index=False)


# **Thanks for reading sofar. Please consider upvoting the kernel if it was usefull.**
