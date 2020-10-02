#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## <center>Traffic Congestion </center>
# ![Traffic Congestion](http://i.4pcdn.org/pol/1407750704541.jpg)
# 
# ### Objective: 
# Predict traffic congestion, based on an aggregate measure of stopping distance and waiting times, at intersections in 4 major US cities: Atlanta, Boston, Chicago & Philadelphia using [BigQuery](https://cloud.google,com/bigquery), a data warehouse for manipulating, joining, and querying large scale tabular datasets. BigQuery also offers [BigQuery ML](https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro), an easy way for users to create and run machine learning models to generate predictions through a SQL query interface.
# 

# In[ ]:


# Set your own project id here
PROJECT_ID = 'geotabintersection'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_geotab', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# The BigQuery client library provides a cell magic, %%bigquery, which runs a SQL query and returns the results as a Pandas DataFrame. Once you use this command the rest of your cell will be treated as a SQL command. (Note that tab complete won't work for SQL code written in this way.)

# In[ ]:


from google.cloud.bigquery.magics import _run_query 
import json
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[ ]:


with open('../input/bigquery-geotab-intersection-congestion/submission_metric_map.json', 'rt') as myfile:
     sub_lab = json.load(myfile)
labels = {val:key for key, val in sub_lab.items()}


# In[ ]:


# create a reference to our table
tr_table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(tr_table, max_results=5).to_dataframe()


# In[ ]:


# create a reference to our table
te_table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.test")

# look at five rows from our dataset
client.list_rows(te_table, max_results=5).to_dataframe()


# In[ ]:


for field in tr_table.schema:
    print(field.name, field.field_type)


# Get Total rows in the training data

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n    COUNT(*) AS totalrowsTrain\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`')


# Get the column names for the training set

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n    *\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId = 0')


# In[ ]:


# create a reference to our table
test = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.test")

# look at five rows from our dataset
client.list_rows(test, max_results=5).to_dataframe()


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT COUNT(*) AS totalrowsTest\nFROM  `kaggle-competition-datasets.geotab_intersection_congestion.test`')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n    *\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.test`\nWHERE RowId = 0')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'Avg_time_df', 'SELECT\n    AVG(TotalTimeStopped_p20) As Avg_t_p20,\n    AVG(TotalTimeStopped_p40) As Avg_t_p40,\n    AVG(TotalTimeStopped_p50) As Avg_t_p50,\n    AVG(TotalTimeStopped_p60) As Avg_t_p60,\n    AVG(TotalTimeStopped_p80) As Avg_t_p80,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGroup By City')


# This means for Atlanta 80percntile of the people have to stop on an average 27.89 min or so and Boston 26.22 . Hope this inference is right

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
Avg_time_df.plot(kind = 'bar' , x ='City' , y = ['Avg_t_p20','Avg_t_p40','Avg_t_p50','Avg_t_p60','Avg_t_p80'],figsize =(12,6))


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'sttudf', "SELECT\n    CASE WHEN (EntryHeading = ExitHeading) THEN 'S' ELSE 'T' END AS straightOrTurn,\n    AVG(TotalTimeStopped_p20) As Avg_t_p20,\n    AVG(TotalTimeStopped_p40) As Avg_t_p40,\n    AVG(TotalTimeStopped_p50) As Avg_t_p50,\n    AVG(TotalTimeStopped_p60) As Avg_t_p60,\n    AVG(TotalTimeStopped_p80) As Avg_t_p80,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGroup By straightOrTurn,City")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))
sttudf.plot(kind='bar', stacked =True, figsize =(15,6))
#sttudf.plot(kind = 'barh' , x ='straightOrTurn', y = ['Avg_t_p20','Avg_t_p40','Avg_t_p50','Avg_t_p60','Avg_t_p80'])


# Just testing creating a stored procedure and executing

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE PROCEDURE bqml_geotab.GetEmpl(r_id INT64, OUT inter_id INT64)\nBEGIN\n  DECLARE cr_rows_id INT64 DEFAULT r_id;\n    SET inter_id = (\n      SELECT IntersectionId FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\n      WHERE RowId = r_id\n    );\nEND;')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'DECLARE r_id INT64 DEFAULT 2079854;\nDECLARE inter_id INT64;\n-- Call the stored procedure to get the hierarchy for this employee ID.\nCALL bqml_geotab.GetEmpl(r_id, inter_id);\n  -- Show the hierarchy for the employee.\nSELECT inter_id;')


# In[ ]:


def make_query(query_text, job_config=None, **kwargs):
    query = _run_query(
        client, query_text.format(**kwargs),
        job_config=job_config)
    return query.to_dataframe()


# In[ ]:


select_q = """
        CAST(IntersectionId AS string) AS IntersectionId,
        ST_GEOHASH(ST_GEOGPOINT(Longitude, Latitude), 10) AS geohash,
        CONCAT(EntryHeading,ExitHeading) AS enexit,
        CASE WHEN (EntryHeading = ExitHeading) THEN 'S' 
        WHEN (EntryHeading = 'N' AND (ExitHeading = 'E' OR ExitHeading = 'NE')) THEN 'R'
        WHEN (EntryHeading = 'N' AND (ExitHeading = 'W' OR ExitHeading = 'NW')) THEN 'L'
        WHEN (EntryHeading = 'S' AND (ExitHeading = 'SW' OR ExitHeading = 'S')) THEN 'R'
        WHEN (EntryHeading = 'S' AND (ExitHeading = 'SE' OR ExitHeading = 'E')) THEN 'L'
        WHEN (EntryHeading = 'E' AND (ExitHeading = 'SE' OR ExitHeading = 'S')) THEN 'R'
        WHEN (EntryHeading = 'E' AND (ExitHeading = 'NE' OR ExitHeading = 'N')) THEN 'L'
        WHEN (EntryHeading = 'W' AND (ExitHeading = 'NW' OR ExitHeading = 'N')) THEN 'R'
        WHEN (EntryHeading = 'W' AND (ExitHeading = 'SW' OR ExitHeading = 'S')) THEN 'L'
        WHEN (EntryHeading = 'NE' AND (ExitHeading = 'E' OR ExitHeading = 'SE')) THEN 'R'
        WHEN (EntryHeading = 'NE' AND (ExitHeading = 'N' OR ExitHeading = 'NW')) THEN 'L'
        WHEN (EntryHeading = 'SE' AND (ExitHeading = 'S' OR ExitHeading = 'SW')) THEN 'R'
        WHEN (EntryHeading = 'SE' AND (ExitHeading = 'E' OR ExitHeading = 'NE')) THEN 'L'
        WHEN (EntryHeading = 'SW' AND (ExitHeading = 'W' OR ExitHeading = 'NW')) THEN 'R'
        WHEN (EntryHeading = 'SW' AND (ExitHeading = 'S' OR ExitHeading = 'SE')) THEN 'L'
        WHEN (EntryHeading = 'NW' AND (ExitHeading = 'N' OR ExitHeading = 'NE')) THEN 'R'
        WHEN (EntryHeading = 'NW' AND (ExitHeading = 'W' OR ExitHeading = 'SW')) THEN 'L'
        ELSE '0' END AS Sorturns,        
        CAST(Hour AS string) AS Hour, 
        CAST(Weekend AS string) AS Weekend,
        CAST(Month AS string) As Month,
        PATH,
        City
    FROM
    """


# In[ ]:


experimental = False
if experimental:
    create_stmt = "CREATE OR REPLACE MODEL"
else: 
    create_stmt = "CREATE MODEL IF NOT EXISTS"
create_model_template = """
{is_experimental} `{model_name}`
    OPTIONS(MODEL_TYPE = 'LINEAR_REG',
    LS_INIT_LEARN_RATE = @init_lr,
            MAX_ITERATIONS = 10 ) AS
SELECT
    {label_name} as label,
    {select_q}
      `kaggle-competition-datasets.geotab_intersection_congestion.train`
WHERE
    RowId < 2600000
    AND {label_name} < @tmp_value
"""


# In[ ]:


#bigquery.ScalarQueryParameter("reg_value", "INT64", 10),
#bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.1), 
#_REG = @rL2eg_value,


# In[ ]:



configs= {
    "bqml_geotab.model_20_0":[   
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 200)
    ],
    "bqml_geotab.model_50_1":[    
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 300)
    ],
    "bqml_geotab.model_80_2":[   
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 500)
    ],
    "bqml_geotab.model_20_3": [    
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 1500)
    ],
    "bqml_geotab.model_50_4": [    
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 2400)
    ],
    "bqml_geotab.model_80_5": [
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.05),
        bigquery.ScalarQueryParameter("tmp_value", "INT64", 3700)
    ]
}


# In[ ]:


for key, value in labels.items():
    #print(key, value)
    lab = key
    labv = lab[-2:]+'_'+value
    model_name = 'bqml_geotab.model_'+labv
    model_name=model_name
    label_name=lab
    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = configs[model_name]
    _ = make_query(create_model_template, 
            job_config=job_config,
            model_name=model_name,
            label_name=label_name,
            select_q = select_q,
            is_experimental=create_stmt)
        
    print(model_name, "is complete")


# In[ ]:


for model in client.list_models('geotabintersection.bqml_geotab'):
    print(model.path)


# In[ ]:


eval_train = """
SELECT
  *
FROM
  ML.TRAINING_INFO(MODEL `{model_name}`) 
ORDER BY iteration 
"""
eval_model="""
SELECT
  *
FROM ML.EVALUATE(MODEL `{model_name}`, (
  SELECT
    {label_name} as label,
    {select_q}
  `kaggle-competition-datasets.geotab_intersection_congestion.train`
WHERE
    RowId >= 2600000
    ))
"""


# In[ ]:


feature2loss = {}
for key, value in labels.items():
    #print(key, value)
    lab = key
    labv = lab[-2:]+'_'+value
    model_name = 'bqml_geotab.model_'+labv
    model_name=model_name
    label_name=lab 
    # evaluating model
    train_info = make_query(
            eval_train,
            model_name=model_name,
            label_name=label_name)
    eval_info = make_query(
            eval_model,
            model_name=model_name,
            label_name=label_name,
            select_q = select_q
        )
    feature2loss[value] = {'eval': eval_info, 
                       'train':train_info.loc[train_info['iteration'].idxmax(),
                                              ['loss', 'eval_loss']]}
    print(value, "train_loss (train, eval)= ",
              *train_info.loc[
            train_info['iteration'].idxmax(),
            ['loss', 'eval_loss']])


# In[ ]:



predict_model="""
SELECT
  RowId,
  predicted_label AS {label_name}
FROM
  ML.PREDICT(MODEL `{model_name}`,
    (
    SELECT
        RowId,
        {select_q}
      `kaggle-competition-datasets.geotab_intersection_congestion.test`))
    ORDER BY RowId ASC
"""


# In[ ]:


def change_columns(df, model_num):
    df['RowId'] = df['RowId'].apply(str) + '_%s'%(model_num)
    df.rename(columns={'RowId': 'TargetId', 
                       sub_lab[model_num]: 'Target'}, 
              inplace=True)


# In[ ]:


results = []
for key, value in labels.items():
    #print(key, value)
    lab = key
    labv = lab[-2:]+'_'+value
    model_name = 'bqml_geotab.model_'+labv
    model_name=model_name
    label_name=lab 
    var = "df_t"+value
    print(var)
    df = make_query(
            predict_model,
            model_name=model_name,
            label_name=label_name,
            select_q = select_q)
    results.append((label_name, df))
    #return results
#results = make_queries(predict_model)


# In[ ]:


predictions = [vframe.copy(deep=True) for _, vframe in results]
keys = [k for k, _ in results]
for k, frame in zip(keys, predictions):
    change_columns(frame, labels[k])
df = pd.concat(predictions)


# In[ ]:


submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')


# In[ ]:


df.to_csv('bqml_submission.csv', index=False)

