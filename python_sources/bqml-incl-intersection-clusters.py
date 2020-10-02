#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# The kernel relies as much as possible on BigQuery (BQ). All features are generated in BQ. And the prediction model is also build in BQ.
# 
# ## Summary
# Features:
# - city, hour, weekend, month
# - 10 clusters of Intersections per city
# - Distance to the nearest cluster
# - 8 directions of turn (left, right, centered, uturn, centered-left, ...)
# - Directions, Entry- and Exit-Heading are embedded and also they are translated into degrees (centered = 90deg; North = 90deg). Afterwards the degree features are split into two features sin(feature_x_deg) and cos(feature_x_deg).
# 
# Model:
# - Linear Regression
# 
# 
# ## Next TODOs
# - reasonable Train-Valid-Split strategy
# 
# ## Credits
# Some of the ideas are inspired by the following kernels. Please visit and give them upvotes if you like them.
# - This kernel is a forked from [BigQuery Machine Learning Tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial).
# - The direction features are like the Flow feature in https://www.kaggle.com/jpmiller/intersection-level-eda

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


mod_names= ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80',
            'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']


# # Cluster Intersections per City

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'city_df', 'SELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.test` t\nUNION DISTINCT\nSELECT t.city\n  FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` t')


# In[ ]:


cities = list(city_df['city'])


# In[ ]:


model_changed = False

if model_changed:
    for c in cities:
        sql="""DROP MODEL `bqml_example.model_cluster_"""+c+"""`"""
        client.query(sql)

        print('Dropped',c)
    
    
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


# ## Checkout the clusters
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


# ## Create city_cluster table

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


model_changed = False

if model_changed:
    sql="DROP TABLE IF EXISTS `bqml_example.city_cluster_train`"
    job_result=client.query(sql).result()
    sql="DROP TABLE IF EXISTS `bqml_example.city_cluster_test`"
    job_result=client.query(sql).result()

    
sql = "CREATE TABLE IF NOT EXISTS `bqml_example.city_cluster_train` as " + feature_sql('ALL','=rowid', True, 'train')
job_result=client.query(sql).result()
sql = "CREATE TABLE IF NOT EXISTS `bqml_example.city_cluster_test` as " + feature_sql('ALL','=rowid', True, 'test')
job_result=client.query(sql).result()
    
    


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', '\nSELECT t.TotalTimeStopped_p20 as label, cc.* except (rowid, TotalTimeStopped_p20,TotalTimeStopped_p50,TotalTimeStopped_p80,DistanceToFirstStop_p20, DistanceToFirstStop_p50, DistanceToFirstStop_p80) \n  FROM `bqml_example.city_cluster_train` cc,\n       `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n WHERE cc.rowid = t.rowid\n LIMIT 20;')


# # Model

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE FUNCTION `bqml_example.direction2degree`(dir string) AS (\n case dir\n   when "C" then\n    90\n   when \'CL\' then\n    135\n   when "L" then\n    180\n   when \'UL\' then\n    225\n   when "U" then\n    270\n   when \'UR\' then\n    315\n   when "R" then\n    0\n   when \'CR\' then\n    45\n    \n   when "N" then\n    90\n   when \'NW\' then\n    135\n   when "W" then\n    180\n   when \'SW\' then\n    225\n   when "S" then\n    270\n   when \'SE\' then\n    315\n   when "E" then\n    0\n   when \'NE\' then\n    45\n end\n);')


# ## Train

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_changed = True\n\nif model_changed:\n    for mn in mod_names:\n        sql="DROP MODEL IF EXISTS `bqml_example.model_"+mn+"`"\n        client.query(sql).result()\n\n        print(\'Drop\',mn)\n\nfor mn in mod_names:\n    \n    sql="""\n    CREATE MODEL IF NOT EXISTS `bqml_example.model_"""+mn+"""`\n    OPTIONS(MODEL_TYPE=\'linear_reg\', \n            L2_REG=0.1,\n            LS_INIT_LEARN_RATE=0.4) AS \n    SELECT  t."""+mn+""" as label,\n            cc.city_cluster,\n            cc.city,\n            cc.hour,\n            cc.weekend,\n            cc.month,\n            cc.direction,\n            cc.entryheading,\n            cc.exitheading,\n            round(sin(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_sin,\n            round(cos(bqml_example.direction2degree(cc.direction)*ACOS(-1)/180),6) direction_cos,\n            round(sin(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_sin,\n            round(cos(bqml_example.direction2degree(cc.entryheading)*ACOS(-1)/180),6) entryheading_cos,\n            round(sin(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_sin,\n            round(cos(bqml_example.direction2degree(cc.exitheading)*ACOS(-1)/180),6) exitheading_cos,\n            round(cc.dist_to_cluster_center,8) dist_to_cluster_center\n      FROM `bqml_example.city_cluster_train` cc,\n           `kaggle-competition-datasets.geotab_intersection_congestion.train` t\n     WHERE t.rowid = cc.rowid\n       AND cc.rowid < 2600000;\n    """\n\n    client.query(sql).result()\n\n    print(\'Done with\',mn)')


# ## Get training statistics
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.model_TotalTimeStopped_p20`)\nORDER BY iteration ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.FEATURE_INFO(MODEL `bqml_example.model_TotalTimeStopped_p20`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.WEIGHTS(MODEL  `bqml_example.model_TotalTimeStopped_p20`,\n    STRUCT(true AS standardize))')


# ## Evaluate your model
# 

# In[ ]:


sql="""SELECT
          *
        FROM ML.EVALUATE(MODEL `bqml_example.model_TotalTimeStopped_p20`, (
        SELECT  t.TotalTimeStopped_p20 as label, 
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
         AND t.rowid > 2600000))"""

client.query(sql).to_dataframe()


# # Predict outcomes
# 

# In[ ]:


def pred(mn, debug=False):
    
    if debug:
        lmt='LIMIT 10'
    else:
        lmt=''
    
    sql="""
    SELECT
      RowId,
      predicted_label as """+mn+"""
    FROM
      ML.PREDICT(MODEL `bqml_example.model_"""+mn+"""`,
        (
        SELECT  cc.RowId, 
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
         WHERE t.rowid = cc.rowid
          """+lmt+"""))
        ORDER BY RowId ASC"""

    return client.query(sql).to_dataframe()
    
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

