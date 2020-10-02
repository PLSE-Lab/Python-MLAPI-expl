#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Replace 'kaggle-competitions-project' with YOUR OWN project id here --  
PROJECT_ID = 'bigquery-geotab'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# create a reference to our table
table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT DISTINCT city FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT DISTINCT city FROM `kaggle-competition-datasets.geotab_intersection_congestion.test`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'WITH train_intersections AS (SELECT COUNT(DISTINCT IntersectionId) AS trainIntersectionCount, City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City),\ntest_intersections AS (SELECT COUNT(DISTINCT IntersectionId) AS testIntersectionCount, City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.test`\nGROUP BY City),\nhelper AS (SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` \n      INTERSECT DISTINCT\n      SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.test`),\nintersections AS (SELECT COUNT(DISTINCT IntersectionId) AS commonIntersectionCount, City FROM helper \n                  GROUP BY City)\nSELECT train_intersections.City, train_intersections.trainIntersectionCount, test_intersections.testIntersectionCount, \n    intersections.commonIntersectionCount\nFROM train_intersections\nINNER JOIN test_intersections \nON train_intersections.City = test_intersections.City\nINNER JOIN intersections\nON train_intersections.City = intersections.City')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT AVG(TotalTimeStopped_p20) AS avg_time_20, MIN(TotalTimeStopped_p20) AS min_time_20,\n    MAX(TotalTimeStopped_p20) AS max_time_20, \n    IFNULL(STDDEV_POP(TotalTimeStopped_p20), 0) AS std_pop_time_20, \n    IFNULL(STDDEV_SAMP(TotalTimeStopped_p20), 0) AS std_samp_time_20, \n    IFNULL(VAR_POP(TotalTimeStopped_p20), 0) AS var_pop_time_20,\n    IFNULL(VAR_SAMP(TotalTimeStopped_p20), 0) AS var_samp_time_20, \n    IFNULL(STDDEV_POP(DISTINCT TotalTimeStopped_p20), 0) AS std_pop_d_time_20, \n    IFNULL(STDDEV_SAMP(DISTINCT TotalTimeStopped_p20), 0) AS std_samp_d_time_20, \n    IFNULL(VAR_POP(DISTINCT TotalTimeStopped_p20), 0) AS var_pop_d_time_20, \n    IFNULL(VAR_SAMP(DISTINCT TotalTimeStopped_p20), 0) AS var_samp_d_time_20,\n    City, IntersectionId\nFROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City, IntersectionId\nLIMIT 10')


# In[ ]:


labels = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 
          'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']
for label in labels:
    label_type = 'FLOAT64' if label.startswith('Distance') else 'INT64'
    schema = [
        bigquery.SchemaField(label, label_type, mode="REQUIRED"),
        bigquery.SchemaField("avg_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("min_stats", label_type, mode="REQUIRED"),
        bigquery.SchemaField("max_stats", label_type, mode="REQUIRED"),
        bigquery.SchemaField("std_pop_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_samp_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_pop_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_samp_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_pop_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_samp_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_pop_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_samp_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("EntryHeading", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ExitHeading", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Weekend", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("Hour", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("Month", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("Path", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("City", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("IntersectionId", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("RowId", "INT64", mode="REQUIRED")
    ]

    dataset_ref = client.dataset('model_dataset')
    table_ref = dataset_ref.table('stats_dataset_' + label)
#     client.delete_table(table_ref)
    table = bigquery.Table('bigquery-geotab.model_dataset.stats_dataset_' + label, schema=schema)
    table = client.create_table(table)

    sql = """
            WITH stats AS (SELECT AVG({label}) AS avg_stats, MIN({label}) AS min_stats,
                MAX({label}) AS max_stats, 
                IFNULL(STDDEV_POP({label}), 0) AS std_pop_stats, 
                IFNULL(STDDEV_SAMP({label}), 0) AS std_samp_stats, 
                IFNULL(VAR_POP({label}), 0) AS var_pop_stats,
                IFNULL(VAR_SAMP({label}), 0) AS var_samp_stats, 
                IFNULL(STDDEV_POP(DISTINCT {label}), 0) AS std_pop_d_stats, 
                IFNULL(STDDEV_SAMP(DISTINCT {label}), 0) AS std_samp_d_stats, 
                IFNULL(VAR_POP(DISTINCT {label}), 0) AS var_pop_d_stats, 
                IFNULL(VAR_SAMP(DISTINCT {label}), 0) AS var_samp_d_stats,
                City, IntersectionId
            FROM
                `kaggle-competition-datasets.geotab_intersection_congestion.train`
            GROUP BY City, IntersectionId)
            SELECT 
                {label},
                avg_stats, min_stats,
                max_stats, 
                std_pop_stats, std_samp_stats, 
                var_pop_stats, var_samp_stats, 
                std_pop_d_stats, 
                std_samp_d_stats, 
                var_pop_d_stats, 
                var_samp_d_stats,
                EntryHeading,
                ExitHeading,
                Weekend,
                Hour,
                Month,
                Path, 
                dataset.City,
                dataset.IntersectionId,
                dataset.RowId
            FROM 
                `kaggle-competition-datasets.geotab_intersection_congestion.train` AS dataset
            INNER JOIN stats
            ON dataset.City = stats.City AND dataset.IntersectionId = stats.IntersectionId
            """.format(label=label)
    job_config = bigquery.QueryJobConfig()
    job_config.destination = table_ref

    query_job = client.query(
        sql,
        # Location must match that of the dataset(s) referenced in the query
        # and of the destination table.
        location='US',
        job_config=job_config)  # API request - starts the query

    query_job.result()


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(TotalTimeStopped_p20, TotalTimeStopped_p20),\n    CORR(TotalTimeStopped_p20, avg_stats) AS avg_stats, \n    CORR(TotalTimeStopped_p20, min_stats) AS min_stats,\n    CORR(TotalTimeStopped_p20, max_stats) AS max_stats, \n    CORR(TotalTimeStopped_p20, std_pop_stats) AS std_pop_stats, \n    CORR(TotalTimeStopped_p20, std_samp_stats) AS std_samp_stats, \n    CORR(TotalTimeStopped_p20, var_pop_stats) AS var_pop_stats, \n    CORR(TotalTimeStopped_p20, var_samp_stats) AS var_samp_stats, \n    CORR(TotalTimeStopped_p20, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(TotalTimeStopped_p20, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(TotalTimeStopped_p20, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(TotalTimeStopped_p20, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(TotalTimeStopped_p20, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(TotalTimeStopped_p20, Hour) AS Hour,\n    CORR(TotalTimeStopped_p20, Month) AS Month,\n    CORR(TotalTimeStopped_p20, IntersectionId) AS IntersectionId,\n    CORR(TotalTimeStopped_p20, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p20`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(TotalTimeStopped_p50, TotalTimeStopped_p50),\n    CORR(TotalTimeStopped_p50, avg_stats) AS avg_stats, \n    CORR(TotalTimeStopped_p50, min_stats) AS min_stats,\n    CORR(TotalTimeStopped_p50, max_stats) AS max_stats, \n    CORR(TotalTimeStopped_p50, std_pop_stats) AS std_pop_stats, \n    CORR(TotalTimeStopped_p50, std_samp_stats) AS std_samp_stats, \n    CORR(TotalTimeStopped_p50, var_pop_stats) AS var_pop_stats, \n    CORR(TotalTimeStopped_p50, var_samp_stats) AS var_samp_stats, \n    CORR(TotalTimeStopped_p50, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(TotalTimeStopped_p50, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(TotalTimeStopped_p50, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(TotalTimeStopped_p50, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(TotalTimeStopped_p50, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(TotalTimeStopped_p50, Hour) AS Hour,\n    CORR(TotalTimeStopped_p50, Month) AS Month,\n    CORR(TotalTimeStopped_p50, IntersectionId) AS IntersectionId,\n    CORR(TotalTimeStopped_p50, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p50`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(TotalTimeStopped_p80, TotalTimeStopped_p80),\n    CORR(TotalTimeStopped_p80, avg_stats) AS avg_stats, \n    CORR(TotalTimeStopped_p80, min_stats) AS min_stats,\n    CORR(TotalTimeStopped_p80, max_stats) AS max_stats, \n    CORR(TotalTimeStopped_p80, std_pop_stats) AS std_pop_stats, \n    CORR(TotalTimeStopped_p80, std_samp_stats) AS std_samp_stats, \n    CORR(TotalTimeStopped_p80, var_pop_stats) AS var_pop_stats, \n    CORR(TotalTimeStopped_p80, var_samp_stats) AS var_samp_stats, \n    CORR(TotalTimeStopped_p80, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(TotalTimeStopped_p80, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(TotalTimeStopped_p80, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(TotalTimeStopped_p80, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(TotalTimeStopped_p80, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(TotalTimeStopped_p80, Hour) AS Hour,\n    CORR(TotalTimeStopped_p80, Month) AS Month,\n    CORR(TotalTimeStopped_p80, IntersectionId) AS IntersectionId,\n    CORR(TotalTimeStopped_p80, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p80`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(DistanceToFirstStop_p20, DistanceToFirstStop_p20),\n    CORR(DistanceToFirstStop_p20, avg_stats) AS avg_stats, \n    CORR(DistanceToFirstStop_p20, min_stats) AS min_stats,\n    CORR(DistanceToFirstStop_p20, max_stats) AS max_stats, \n    CORR(DistanceToFirstStop_p20, std_pop_stats) AS std_pop_stats, \n    CORR(DistanceToFirstStop_p20, std_samp_stats) AS std_samp_stats, \n    CORR(DistanceToFirstStop_p20, var_pop_stats) AS var_pop_stats, \n    CORR(DistanceToFirstStop_p20, var_samp_stats) AS var_samp_stats, \n    CORR(DistanceToFirstStop_p20, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(DistanceToFirstStop_p20, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(DistanceToFirstStop_p20, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(DistanceToFirstStop_p20, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(DistanceToFirstStop_p20, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(DistanceToFirstStop_p20, Hour) AS Hour,\n    CORR(DistanceToFirstStop_p20, Month) AS Month,\n    CORR(DistanceToFirstStop_p20, IntersectionId) AS IntersectionId,\n    CORR(DistanceToFirstStop_p20, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p20`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(DistanceToFirstStop_p50, DistanceToFirstStop_p50),\n    CORR(DistanceToFirstStop_p50, avg_stats) AS avg_stats, \n    CORR(DistanceToFirstStop_p50, min_stats) AS min_stats,\n    CORR(DistanceToFirstStop_p50, max_stats) AS max_stats, \n    CORR(DistanceToFirstStop_p50, std_pop_stats) AS std_pop_stats, \n    CORR(DistanceToFirstStop_p50, std_samp_stats) AS std_samp_stats, \n    CORR(DistanceToFirstStop_p50, var_pop_stats) AS var_pop_stats, \n    CORR(DistanceToFirstStop_p50, var_samp_stats) AS var_samp_stats, \n    CORR(DistanceToFirstStop_p50, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(DistanceToFirstStop_p50, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(DistanceToFirstStop_p50, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(DistanceToFirstStop_p50, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(DistanceToFirstStop_p50, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(DistanceToFirstStop_p50, Hour) AS Hour,\n    CORR(DistanceToFirstStop_p50, Month) AS Month,\n    CORR(DistanceToFirstStop_p50, IntersectionId) AS IntersectionId,\n    CORR(DistanceToFirstStop_p50, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p50`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CORR(DistanceToFirstStop_p80, DistanceToFirstStop_p80),\n    CORR(DistanceToFirstStop_p80, avg_stats) AS avg_stats, \n    CORR(DistanceToFirstStop_p80, min_stats) AS min_stats,\n    CORR(DistanceToFirstStop_p80, max_stats) AS max_stats, \n    CORR(DistanceToFirstStop_p80, std_pop_stats) AS std_pop_stats, \n    CORR(DistanceToFirstStop_p80, std_samp_stats) AS std_samp_stats, \n    CORR(DistanceToFirstStop_p80, var_pop_stats) AS var_pop_stats, \n    CORR(DistanceToFirstStop_p80, var_samp_stats) AS var_samp_stats, \n    CORR(DistanceToFirstStop_p80, std_pop_d_stats) AS std_pop_d_stats, \n    CORR(DistanceToFirstStop_p80, std_samp_d_stats) AS std_samp_d_stats, \n    CORR(DistanceToFirstStop_p80, var_pop_d_stats) AS var_pop_d_stats, \n    CORR(DistanceToFirstStop_p80, var_samp_d_stats) AS var_samp_d_stats,\n    CORR(DistanceToFirstStop_p80, CAST(Weekend AS INT64)) AS Weekend,\n    CORR(DistanceToFirstStop_p80, Hour) AS Hour,\n    CORR(DistanceToFirstStop_p80, Month) AS Month,\n    CORR(DistanceToFirstStop_p80, IntersectionId) AS IntersectionId,\n    CORR(DistanceToFirstStop_p80, RowId) AS RowId\nFROM\n    `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p80`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.sample_model`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p20 AS label,\n    avg_stats, \n#     min_stats,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     IntersectionId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p20`\nWHERE\n    RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `model_dataset.sample_model`)\nORDER BY iteration ')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `model_dataset.sample_model`, (\nSELECT\n    TotalTimeStopped_p20 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     IntersectionId,\n#     RowId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p20`\nWHERE\n    RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.TotalTimeStopped_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p20 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     IntersectionId\n#     RowId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p20`")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.TotalTimeStopped_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p50 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     IntersectionId\n#     RowId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p50`")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.TotalTimeStopped_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p80 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n    Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City,\n    RowId\n#     IntersectionId\n#     RowId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_TotalTimeStopped_p80`")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.DistanceToFirstStop_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p20 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     RowId\n#     IntersectionId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p20`")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.DistanceToFirstStop_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p50 AS label,\n    avg_stats, \n#     min_time_20,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     RowId\n#     IntersectionId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p50`")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.DistanceToFirstStop_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p80 AS label,\n    avg_stats, \n    min_stats,\n    max_stats, \n    std_pop_stats, std_samp_stats, \n    var_pop_stats, var_samp_stats, \n    std_pop_d_stats, \n    std_samp_d_stats, \n    var_pop_d_stats, \n    var_samp_d_stats,\n    EntryHeading,\n    ExitHeading,\n#     Weekend,\n#     Hour,\n#     Month,\n    Path, \n    City\n#     RowId\n#     IntersectionId\nFROM\n  `bigquery-geotab.model_dataset.stats_dataset_DistanceToFirstStop_p80`")


# In[ ]:


labels = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 
          'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']
for label in labels:
    label_type = 'FLOAT64' if label.startswith('Distance') else 'INT64'
    schema = [
        bigquery.SchemaField("avg_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("min_stats", label_type, mode="REQUIRED"),
        bigquery.SchemaField("max_stats", label_type, mode="REQUIRED"),
        bigquery.SchemaField("std_pop_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_samp_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_pop_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_samp_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_pop_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("std_samp_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_pop_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("var_samp_d_stats", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("EntryHeading", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ExitHeading", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("Weekend", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("Hour", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("Month", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("Path", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("City", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("IntersectionId", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("RowId", "INT64", mode="REQUIRED")
    ]

    dataset_ref = client.dataset('model_dataset')
    table_ref = dataset_ref.table('test_stats_dataset_' + label)
#     client.delete_table(table_ref)
    table = bigquery.Table('bigquery-geotab-test-dataset.model_dataset.test_stats_dataset_' + label, schema=schema)
    table = client.create_table(table)

    sql = """
            WITH remaining_test_dataset AS (
                SELECT
                    test_dataset.City,
                    test_dataset.IntersectionId,
                    MIN(Longitude) AS Longitude,
                    MIN(Latitude) AS Latitude
                FROM 
                    `kaggle-competition-datasets.geotab_intersection_congestion.test` AS test_dataset
                INNER JOIN (
                    SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.test` 
                    EXCEPT DISTINCT
                    SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`
                ) AS except_intersections
                ON test_dataset.City = except_intersections.City AND test_dataset.IntersectionId = except_intersections.IntersectionId
                GROUP BY test_dataset.City, test_dataset.IntersectionId
            ), 
            train_helper AS (
                SELECT 
                    IntersectionId, 
                    City, 
                    MIN(Longitude) AS Longitude, 
                    MIN(Latitude) AS Latitude
                FROM 
                    `kaggle-competition-datasets.geotab_intersection_congestion.train`
                GROUP BY IntersectionId, City
            ),
            nearest_remaining_test_dataset AS (    
                SELECT AS VALUE ARRAY_AGG(STRUCT<IntersectionId INT64, City STRING, 
                                          NearestIntersectionId INT64, NearestCity STRING>(test.IntersectionId, test.City, 
                                                                                           train.IntersectionId, train.City) 
                                          ORDER BY ST_DISTANCE(test.point, train.point) LIMIT 1)[OFFSET(0)] 
                FROM (SELECT IntersectionId, City, ST_GEOGPOINT(Longitude, Latitude) point FROM remaining_test_dataset) test
                CROSS JOIN (SELECT IntersectionId, City, ST_GEOGPOINT(Longitude, Latitude) point FROM train_helper) train 
                GROUP BY test.IntersectionId, test.City
            )
            SELECT 
                avg_stats, min_stats,
                max_stats, 
                std_pop_stats, std_samp_stats, 
                var_pop_stats, var_samp_stats, 
                std_pop_d_stats, 
                std_samp_d_stats, 
                var_pop_d_stats, 
                var_samp_d_stats,
                test_dataset.EntryHeading,
                test_dataset.ExitHeading,
                test_dataset.Weekend,
                test_dataset.Hour,
                test_dataset.Month,
                test_dataset.Path, 
                test_dataset.City,
                test_dataset.IntersectionId,
                test_dataset.RowId
            FROM 
                `kaggle-competition-datasets.geotab_intersection_congestion.test` AS test_dataset
            INNER JOIN nearest_remaining_test_dataset
                ON test_dataset.City = nearest_remaining_test_dataset.City AND test_dataset.IntersectionId = nearest_remaining_test_dataset.IntersectionId
            INNER JOIN `bigquery-geotab.model_dataset.stats_dataset_{label}` as stats_dataset
                ON nearest_remaining_test_dataset.NearestCity = stats_dataset.City AND nearest_remaining_test_dataset.NearestIntersectionId = stats_dataset.IntersectionId 
            
            UNION ALL
            
            SELECT 
                avg_stats, min_stats,
                max_stats, 
                std_pop_stats, std_samp_stats, 
                var_pop_stats, var_samp_stats, 
                std_pop_d_stats, 
                std_samp_d_stats, 
                var_pop_d_stats, 
                var_samp_d_stats,
                test_dataset.EntryHeading,
                test_dataset.ExitHeading,
                test_dataset.Weekend,
                test_dataset.Hour,
                test_dataset.Month,
                test_dataset.Path, 
                test_dataset.City,
                test_dataset.IntersectionId,
                test_dataset.RowId
            FROM 
                `kaggle-competition-datasets.geotab_intersection_congestion.test` AS test_dataset
            INNER JOIN (
                SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.train` 
                INTERSECT DISTINCT
                SELECT IntersectionId, City FROM `kaggle-competition-datasets.geotab_intersection_congestion.test`
            ) AS common_intersections
            ON test_dataset.City = common_intersections.City 
                AND test_dataset.IntersectionId = common_intersections.IntersectionId
            INNER JOIN `bigquery-geotab.model_dataset.stats_dataset_{label}` as stats_dataset
            ON stats_dataset.City = test_dataset.City 
                    AND stats_dataset.IntersectionId = test_dataset.IntersectionId 
            """.format(label=label)
    job_config = bigquery.QueryJobConfig()
    job_config.destination = table_ref

    query_job = client.query(
        sql,
        # Location must match that of the dataset(s) referenced in the query
        # and of the destination table.
        location='US',
        job_config=job_config)  # API request - starts the query

    query_job.result()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'TotalTimeStopped_p20_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.TotalTimeStopped_p20`,\n    (\n    SELECT\n        avg_stats, \n    #     min_time_20,\n        max_stats, \n        std_pop_stats, std_samp_stats, \n        var_pop_stats, var_samp_stats, \n        std_pop_d_stats, \n        std_samp_d_stats, \n        var_pop_d_stats, \n        var_samp_d_stats,\n        EntryHeading,\n        ExitHeading,\n    #     Weekend,\n    #     Hour,\n    #     Month,\n        Path, \n        City\n    #     IntersectionId\n    #     RowId\n    FROM\n      `bigquery-geotab-test-dataset.model_dataset.test_stats_dataset_TotalTimeStopped_p20`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'TotalTimeStopped_p50_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.TotalTimeStopped_p50`,\n    (\n    SELECT\n        avg_stats, \n    #     min_time_20,\n        max_stats, \n        std_pop_stats, std_samp_stats, \n        var_pop_stats, var_samp_stats, \n        std_pop_d_stats, \n        std_samp_d_stats, \n        var_pop_d_stats, \n        var_samp_d_stats,\n        EntryHeading,\n        ExitHeading,\n    #     Weekend,\n    #     Hour,\n    #     Month,\n        Path, \n        City\n    #     IntersectionId\n    #     RowId\n    FROM\n      `bigquery-geotab.model_dataset.test_stats_dataset_TotalTimeStopped_p50`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'TotalTimeStopped_p80_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.TotalTimeStopped_p80`,\n    (\n        SELECT\n            avg_stats, \n        #     min_time_20,\n            max_stats, \n            std_pop_stats, std_samp_stats, \n            var_pop_stats, var_samp_stats, \n            std_pop_d_stats, \n            std_samp_d_stats, \n            var_pop_d_stats, \n            var_samp_d_stats,\n            EntryHeading,\n            ExitHeading,\n            Weekend,\n        #     Hour,\n        #     Month,\n            Path, \n            City,\n            RowId\n        #     IntersectionId\n        #     RowId\n        FROM\n          `bigquery-geotab.model_dataset.test_stats_dataset_TotalTimeStopped_p80`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'DistanceToFirstStop_p20_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.DistanceToFirstStop_p20`,\n    (\n        SELECT\n            avg_stats, \n        #     min_time_20,\n            max_stats, \n            std_pop_stats, std_samp_stats, \n            var_pop_stats, var_samp_stats, \n            std_pop_d_stats, \n            std_samp_d_stats, \n            var_pop_d_stats, \n            var_samp_d_stats,\n            EntryHeading,\n            ExitHeading,\n        #     Weekend,\n        #     Hour,\n        #     Month,\n            Path, \n            City\n        #     RowId\n        #     IntersectionId\n        FROM\n          `bigquery-geotab.model_dataset.test_stats_dataset_DistanceToFirstStop_p20`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'DistanceToFirstStop_p50_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.DistanceToFirstStop_p50`,\n    (\n        SELECT\n            avg_stats, \n        #     min_time_20,\n            max_stats, \n            std_pop_stats, std_samp_stats, \n            var_pop_stats, var_samp_stats, \n            std_pop_d_stats, \n            std_samp_d_stats, \n            var_pop_d_stats, \n            var_samp_d_stats,\n            EntryHeading,\n            ExitHeading,\n        #     Weekend,\n        #     Hour,\n        #     Month,\n            Path, \n            City\n        #     RowId\n        #     IntersectionId\n        FROM\n          `bigquery-geotab.model_dataset.test_stats_dataset_DistanceToFirstStop_p50`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'DistanceToFirstStop_p80_df', 'SELECT\n  RowId, predicted_label\nFROM\n  ML.PREDICT(MODEL `model_dataset.DistanceToFirstStop_p80`,\n    (\n        SELECT\n            DistanceToFirstStop_p80 AS label,\n            avg_stats, \n            min_stats,\n            max_stats, \n            std_pop_stats, std_samp_stats, \n            var_pop_stats, var_samp_stats, \n            std_pop_d_stats, \n            std_samp_d_stats, \n            var_pop_d_stats, \n            var_samp_d_stats,\n            EntryHeading,\n            ExitHeading,\n        #     Weekend,\n        #     Hour,\n        #     Month,\n            Path, \n            City\n        #     RowId\n        #     IntersectionId\n        FROM\n          `bigquery-geotab.model_dataset.test_stats_dataset_DistanceToFirstStop_p80`))\n    ORDER BY RowId ASC')


# In[ ]:


TotalTimeStopped_p20_df['RowId'] = TotalTimeStopped_p20_df['RowId'].apply(str) + '_0'
TotalTimeStopped_p50_df['RowId'] = TotalTimeStopped_p50_df['RowId'].apply(str) + '_1'
TotalTimeStopped_p80_df['RowId'] = TotalTimeStopped_p80_df['RowId'].apply(str) + '_2'
DistanceToFirstStop_p20_df['RowId'] = DistanceToFirstStop_p20_df['RowId'].apply(str) + '_3'
DistanceToFirstStop_p50_df['RowId'] = DistanceToFirstStop_p50_df['RowId'].apply(str) + '_4'
DistanceToFirstStop_p80_df['RowId'] = DistanceToFirstStop_p80_df['RowId'].apply(str) + '_5'
df = pd.concat([TotalTimeStopped_p20_df, TotalTimeStopped_p50_df, TotalTimeStopped_p80_df, 
                DistanceToFirstStop_p20_df, DistanceToFirstStop_p50_df, DistanceToFirstStop_p80_df])
df.rename(columns={'RowId': 'TargetId', 'predicted_label': 'Target'}, inplace=True)
df


# In[ ]:


df.to_csv(r'submission.csv')

