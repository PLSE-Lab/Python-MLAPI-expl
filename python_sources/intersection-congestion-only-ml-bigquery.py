#!/usr/bin/env python
# coding: utf-8

# This Notebook presents how to make an ML model only with BigQuery ML tools
#  
# For this I have left the notebook 
# 
# https://www.kaggle.com/sirtorry/bigquery-ml-template-intersection-congestion

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Set your own project id here
PROJECT_ID = 'kagglegeotab'

from google.cloud import bigquery
bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# The BigQuery Python client library provides a magic command that allows you to run queries with minimal code.
# 
# * %load_ext google.cloud.bigquery
# 
# The BigQuery client library provides a cell magic, %%bigquery, which runs a SQL query and returns the results as a Pandas DataFrame
# 
# * %%bigquery
# SQL query
# 

# In[ ]:


# magic command
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


#  ## Exploratory Data Analysis

# DATA TEST

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT *\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.test`\nORDER BY RowId\nLIMIT 10')


# In[ ]:


get_ipython().run_cell_magic('bigquery', ' ', "SELECT column_name, data_type, is_nullable \nFROM\n `kaggle-competition-datasets`.geotab_intersection_congestion.INFORMATION_SCHEMA.COLUMNS\nWHERE table_name = 'test'")


# In[ ]:


get_ipython().run_cell_magic('bigquery', ' ', 'SELECT\n    COUNT(*) as NRow\n    , MIN(RowId) as MinRow    \n    , MAX(RowId) as MaxRowId\n    , COUNT(distinct IntersectionId) AS NumIntersectionId\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.test`')


# #### Missing Data Test
# 
# We generate an SQL statement based on the information in the table schema, where a count (*) - count (field name) is made for each field, this difference is the number of undefined values.

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' sqlMissingData', "SELECT CONCAT('SELECT '\n  , STRING_AGG(texto,',')\n  , ' FROM '\n  , ' `kaggle-competition-datasets.geotab_intersection_congestion.test`') as strSqlMissingDataResumen\nFROM \n  (SELECT CONCAT(' count(*) - count(', column_name, ') as NNull_', column_name) as Texto\n  FROM\n   `kaggle-competition-datasets`.geotab_intersection_congestion.INFORMATION_SCHEMA.COLUMNS\n  WHERE table_name = 'test') \n")


# We execute the sentence generated

# In[ ]:


bigquery_client.query(sqlMissingData.iat[0,0]).to_dataframe().stack()


# TRAIN

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT *\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nORDER BY RowId\nLIMIT 10')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "SELECT column_name, data_type, is_nullable \nFROM\n `kaggle-competition-datasets`.geotab_intersection_congestion.INFORMATION_SCHEMA.COLUMNS\nWHERE table_name = 'train'")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n    COUNT(*) as NRow\n    , MIN(RowId) as MinRow    \n    , MAX(RowId) as MaxRowId\n    , COUNT(distinct IntersectionId) AS NumIntersectionId\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`')


# #### Missing Data Train

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' sqlMissingData', "SELECT CONCAT('SELECT '\n  , STRING_AGG(texto,',')\n  , ' FROM '\n  , ' `kaggle-competition-datasets.geotab_intersection_congestion.train`') as strSqlMissingDataResumen\nFROM \n  (SELECT CONCAT(' count(*) - count(', column_name, ') as NNull_', column_name) as Texto\n  FROM\n   `kaggle-competition-datasets`.geotab_intersection_congestion.INFORMATION_SCHEMA.COLUMNS\n  WHERE table_name = 'train') y")


# In[ ]:


bigquery_client.query(sqlMissingData.iat[0,0]).to_dataframe().stack()


# ### Visualizing the data

# Grouping by cities and origin (train or test) of the number of intersections

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' CountIntersectionIdCity', 'SELECT\n    CAST(1 AS BOOL) as IndTrain,\n    City,\n    COUNT(IntersectionId) AS NumIntersectionId\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nUNION ALL\nSELECT\n    CAST(0 AS BOOL)  as IndTrain,\n    City,\n    COUNT(IntersectionId) AS NumIntersectionId\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.test`\nGROUP BY City\nORDER BY City')


# In[ ]:



sns.barplot(x="City", y="NumIntersectionId", hue="IndTrain", data=CountIntersectionIdCity)


# Grouping by city and hours, on training set

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' CountRowByHourCityTrain', 'SELECT\n    City,\n    Hour,\n    COUNT(*) AS NRow\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City, Hour\nORDER BY City, Hour')


# In[ ]:


sns.barplot(x="Hour", y="NRow", hue="City", data=CountRowByHourCityTrain)


# Grouping by city and hours, on test set

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' CountRowByHourCityTest', 'SELECT\n    City,\n    Hour,\n    COUNT(*) AS NRow\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.test`\nGROUP BY City, Hour\nORDER BY City, Hour')


# In[ ]:


sns.barplot(x="Hour", y="NRow", hue="City", data=CountRowByHourCityTest)


# Grouping by city and month, on training set

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' CountRowByMonthCityTrain', 'SELECT\n    City,\n    Month,\n    COUNT(*) AS NRow,\n    AVG(TotalTimeStopped_p50) AS MeanTotalTimeStopped_p50\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City, Month\nORDER BY City, Month')


# In[ ]:



fig, ax =plt.subplots(1,2)
sns.barplot(x="Month", y="NRow", hue="City", data=CountRowByMonthCityTrain, ax=ax[1])
sns.barplot(x="Month", y="MeanTotalTimeStopped_p50", hue="City", data=CountRowByMonthCityTrain, ax=ax[0])
fig.set_size_inches(15, 6)
fig.show()


# Grouping by city and percentile, on training set

# In[ ]:


get_ipython().run_cell_magic('bigquery', ' MeanTimeStoppedTrain', 'SELECT\n    City,\n    20 as p,\n    AVG(TotalTimeStopped_p20) AS MeanTotalTimeStopped\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nunion all \nSELECT\n    City,\n    50 as p,\n    AVG(TotalTimeStopped_p50) AS MeanTotalTimeStopped\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nunion all\nSELECT\n    City,\n    80 as p,\n    AVG(TotalTimeStopped_p80) AS MeanTotalTimeStopped\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City')


# In[ ]:


sns.barplot(x="p", y="MeanTotalTimeStopped", hue="City", data=MeanTimeStoppedTrain)


# ## NEW DATASET
# 
# We generate a new dataset 'kaggleCompetitionWorkDatasets' in BigQuery about our project in order to save data and models. 

# In[ ]:


dataset = bigquery_client.create_dataset('kaggleCompetitionWorkDatasets', exists_ok=True)


# ## SEPARATE TEST SET
# 
# We separated the training set into two parts, to later be able to perform model evaluations. Being such a large set of training we will select 10% of the data to carry out the evaluations.
# 
# For this we generate the fingerprint with the possible predictors, this operation will generate a number and on this we will calculate the module 100 of the absolute value, with this value we will take the ones under 10 and save the RowId of these rows in a table.
# 
# The result is a table with approximately 10% of the identifiers, from this table we will generate another with the rest of the data.

# In[ ]:


job_config = bigquery.QueryJobConfig()
tbTrainTest = dataset.table('train_test')
job_config.destination = tbTrainTest
sql = """
    SELECT RowId
    FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`
    where MOD(ABS(
        FARM_FINGERPRINT(
          CONCAT(
            City
            , CAST(Latitude AS STRING)
            , CAST(Longitude AS STRING)
            , CAST(IntersectionId AS STRING)
            , EntryStreetName
            , ExitStreetName
            , EntryHeading
            , ExitHeading
            , CAST(month AS STRING)
            , CAST(Hour AS STRING)
            , CAST(weekend AS STRING)
            )
        )
      ),100) < 10;
"""

# Start the query, passing in the extra configuration.
query_job = bigquery_client.query(
    sql,
    # Location must match that of the dataset(s) referenced in the query
    # and of the destination table.
    location='US',
    job_config=job_config)  # API request - starts the query

query_job.result()  # Waits for the query to finish

print('Query results loaded to table {}'.format(tbTrainTest.path))


# In[ ]:


job_config = bigquery.QueryJobConfig()
tbTrainTrain = dataset.table('train_train')
job_config.destination = tbTrainTrain
sql = """
    SELECT
      train.RowId
    FROM
      `kaggle-competition-datasets.geotab_intersection_congestion.train` train
    LEFT JOIN
      `kaggleCompetitionWorkDatasets.train_test` train_test
    ON
      train.Rowid = train_test.RowId
    WHERE
      train_test.RowId IS NULL
"""

# Start the query, passing in the extra configuration.
query_job = bigquery_client.query(
    sql,
    # Location must match that of the dataset(s) referenced in the query
    # and of the destination table.
    location='US',
    job_config=job_config)  # API request - starts the query

query_job.result()  # Waits for the query to finish

print('Query results loaded to table {}'.format(tbTrainTrain.path))


# ## Creation of the models
# 
# To create the models, we will use a list with the names of the tags to predict and we will create a function that will receive the tag as a parameter and create the sql script for the creation of the model. The model name will be "model_LR_" + the name of the label.
# 
# The model script will be:
# 
# >CREATE MODEL IF NOT EXISTS `kaggleCompetitionWorkDatasets.model_LR_xxx` <br> 
#   &nbsp;&nbsp;TRANSFORM (label, <br> 
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ML.QUANTILE_BUCKETIZE(Hour, 6) OVER() as BHour,  <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Weekend,  <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CONCAT(City, " ", Path)  as PathAmpli)  <br>
#   &nbsp;&nbsp;OPTIONS( MODEL_TYPE ='LINEAR_REG'  <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,OPTIMIZE_STRATEGY = 'AUTO_STRATEGY'  <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,DATA_SPLIT_METHOD = 'AUTO_SPLIT' <br>
#   &nbsp;&nbsp;) AS <br>
# &nbsp;&nbsp;SELECT xxx AS label, <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;Hour, <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;Weekend, <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;City, <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;Path <br>
# &nbsp;&nbsp;FROM <br>
#   &nbsp;&nbsp;&nbsp;&nbsp;`kaggle-competition-datasets.geotab_intersection_congestion.train` train <br>
# &nbsp;&nbsp;INNER JOIN <br>
#   &nbsp;&nbsp;&nbsp;&nbsp;`kaggleCompetitionWorkDatasets.train_train` train_train <br>
# &nbsp;&nbsp;ON <br>
#   &nbsp;&nbsp;&nbsp;&nbsp;train.RowId = train_train.RowId <br>
# 
# 

# In[ ]:


labelModels = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80',
              'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']


# In[ ]:


def sql_create_model(labelModel):
    sql = "CREATE MODEL IF NOT EXISTS `kaggleCompetitionWorkDatasets.model_LR_" + labelModel + "`" + "\n"
    sql += "  TRANSFORM (label, " + "\n"
    sql += "    ML.QUANTILE_BUCKETIZE(Hour, 6) OVER() as BHour, " + "\n"
    sql += "    Weekend, " + "\n"
    sql += """   CONCAT(City, " ", Path)  as PathAmpli) """ + "\n"
    sql += "  OPTIONS( MODEL_TYPE ='LINEAR_REG' " + "\n"
    sql += "    ,OPTIMIZE_STRATEGY = 'AUTO_STRATEGY' " + "\n"
    sql += "    ,DATA_SPLIT_METHOD = 'AUTO_SPLIT' " + "\n"
    sql += "  ) AS " + "\n"
    sql += "SELECT " + labelModel + " AS label, " + "\n"
    sql += "    Hour, " + "\n"
    sql += "    Weekend, " + "\n"
    sql += "    City, " + "\n" 
    sql += "    Path " + "\n"
    sql += "FROM " + "\n"
    sql += "  `kaggle-competition-datasets.geotab_intersection_congestion.train` train " + "\n"
    sql += "INNER JOIN " + "\n"
    sql += "  `kaggleCompetitionWorkDatasets.train_train` train_train " + "\n"
    sql += "ON " + "\n"
    sql += "  train.RowId = train_train.RowId " + "\n"
        
    return sql


# For each label in the list a job will be generated that launches its creation script.

# In[ ]:


for labelModel in labelModels:
  job_config = bigquery.QueryJobConfig()
  sql = sql_create_model(labelModel)
  query_job = bigquery_client.query(sql,location='US',job_config=job_config)  
  query_job.result()  # Waits for the query to finish


# In[ ]:


for model in bigquery_client.list_models('kagglegeotab.kaggleCompetitionWorkDatasets'):
    print(model.path)


# ## EVALUATION OF THE MODELS
# 
# For each model we will obtain the metrics of the training carried out and evaluate it on the test set that was separated from the global training set.

# Training metrics
# 
# The different results of each model will be saved in a 'results_train' table, for this an SQL query will be executed with the union of the following select for each model
# 
# > select `xxx` as model, * <br/> 
# > from ml.evaluate(model `xxx`);
# 
# 

# In[ ]:


job_config = bigquery.QueryJobConfig()
tbEvaluateTrain = dataset.table('results_train')
job_config.destination = tbEvaluateTrain

sql=""
for labelModel in labelModels:
    sql += """select "model_LR_""" + labelModel + """" as model, * """ + "\n"
    sql += "from ml.evaluate(model `kaggleCompetitionWorkDatasets.model_LR_" + labelModel + "`)" + "\n"
    sql += "union all " + "\n"
sql = sql[:-12]

# Start the query, passing in the extra configuration.
query_job = bigquery_client.query(
    sql,
    location='US',
    job_config=job_config)  # API request - starts the query


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM `kaggleCompetitionWorkDatasets.results_train`')


# Test metrics
# 
# As in the previous case, the different results of each model will be saved in a 'results_test' table, an SQL query will also be executed with the union of the execution of the model on the test sets.
# 
# > select `xxx` as model, * <br>
# FROM ML.EVALUATE(MODEL `xxx`, <br> 
# &nbsp;&nbsp;(SELECT <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;TotalTimeStopped_p20 AS label, <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;Hour, <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;Weekend, <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;Month, <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;Path, <br>
#       &nbsp;&nbsp;&nbsp;&nbsp;City <br>
# &nbsp;&nbsp;FROM <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;`kaggle-competition-datasets.geotab_intersection_congestion.train` train <br>
# &nbsp;&nbsp;INNER JOIN <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;`geotab.train_train` train_test <br>
# &nbsp;&nbsp;ON <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;train.RowId = train_test.RowId) <br>
# ) <br>
# 

# In this case we generate a function

# In[ ]:


def sql_evaluate_model(labelModel):
    sql = """select "model_LR_""" + labelModel + """" as model, * """ + "\n"
    sql += "from ml.evaluate(model `kaggleCompetitionWorkDatasets.model_LR_" + labelModel + "`," + "\n"
    sql += "(" 
    sql += "SELECT " + labelModel + " AS label, " + "\n"
    sql += "    Hour, " + "\n"
    sql += "    Weekend, " + "\n"
    sql += "    City, " + "\n" 
    sql += "    Path " + "\n"
    sql += "FROM " + "\n"
    sql += "  `kaggle-competition-datasets.geotab_intersection_congestion.train` train " + "\n"
    sql += "INNER JOIN " + "\n"
    sql += "  `kaggleCompetitionWorkDatasets.train_test` train_test " + "\n"
    sql += "ON " + "\n"
    sql += "  train.RowId = train_test.RowId " + "\n"
    sql += "))" + "\n"

    return sql


# In[ ]:



tbEvaluateTest = dataset.table('results_test')

sql=""
for labelModel in labelModels:
    sql += sql_evaluate_model(labelModel)
    sql += "union all " + "\n"
sql = sql[:-12]

job_config = bigquery.QueryJobConfig()
job_config.destination = tbEvaluateTest
query_job = bigquery_client.query(sql,location='US',job_config=job_config)  
query_job.result()  # Waits for the query to finish


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM `kaggleCompetitionWorkDatasets.results_test`')


# ### PREDICTIONS
# For each model we calculate the predictions, for this a function will be generated that passing the name of the tag to be predicted will generate a sql string by selecting the data from the test set and the model for the tag to be predicted.
# 
# Subsequently a job is launched with the union of all models and saved in a table called results_submission

# In[ ]:


def sql_predice_model(labelModel):
    sql = """SELECT "model_LR_""" + labelModel + """" as model, * """ + "\n"
    sql += "FROM ML.PREDICT(model `kaggleCompetitionWorkDatasets.model_LR_" + labelModel + "`," + "\n"
    sql += "(" 
    sql += "SELECT RowId, " + "\n"
    sql += "    Hour, " + "\n"
    sql += "    Weekend, " + "\n"
    sql += "    City, " + "\n" 
    sql += "    Path " + "\n"
    sql += "FROM " + "\n"
    sql += "  `kaggle-competition-datasets.geotab_intersection_congestion.test`))" + "\n"

    return sql


# In[ ]:


tbSubmission = dataset.table('results_submission')

sql=""
for labelModel in labelModels:
    sql += sql_predice_model(labelModel)
    sql += "union all " + "\n"
sql = sql[:-12]

job_config = bigquery.QueryJobConfig()
job_config.destination = tbSubmission
query_job = bigquery_client.query(sql,location='US',job_config=job_config)  
query_job.result()  # Waits for the query to finish


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * \nFROM `kaggleCompetitionWorkDatasets.results_submission`\nLIMIT 10')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT model, count(*) \nFROM `kaggleCompetitionWorkDatasets.results_submission`\ngroup by model')


# ### Generate results file
# 
# We format the results with a table that joins the model name to the suffix that must be generated for TargetId.
# 
# Negative predictions are left at 0
# 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'submission', 'WITH modelsCod AS (\n  SELECT *\n  FROM UNNEST(\n     [STRUCT("0" as cod, "model_LR_TotalTimeStopped_p20" as model),\n     STRUCT("1" as cod, "model_LR_TotalTimeStopped_p50" as model),\n     STRUCT("2" as cod, "model_LR_TotalTimeStopped_p80" as model),\n     STRUCT("3" as cod, "model_LR_DistanceToFirstStop_p20" as model),\n     STRUCT("4" as cod, "model_LR_DistanceToFirstStop_p50" as model),\n     STRUCT("5" as cod, "model_LR_DistanceToFirstStop_p80" as model)]\n  ) \n)\nSELECT CONCAT(cast(results_submission.RowId as STRING), "_", modelsCod.cod) as TargetId\n  , IF(results_submission.predicted_label<0,0,results_submission.predicted_label) as Target\nFROM `kaggleCompetitionWorkDatasets.results_submission` results_submission\n  INNER JOIN modelsCod\n    ON modelsCod.model = results_submission.model\nORDER BY results_submission.RowId\n  , modelsCod.cod\n\n    ')


# In[ ]:


print(submission.count())
submission.head(20)


# In[ ]:


submission.to_csv('submission.csv', index=False)

