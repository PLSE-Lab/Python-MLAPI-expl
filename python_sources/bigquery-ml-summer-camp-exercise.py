#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# my project id here
PROJECT_ID = 'consummate-web-251305'

# create a client instance for the project
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

print("Project initialized...")


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')

print("Done loading GCP BQML...")


# In[ ]:


# 1.


# ##Use of training vs test data:
# A part of data is used for building and training the ML model on. The remaining part of the data is used for testing the model we just built. Usually this division is 9/10 and 1/10 parts respectively, or 4/5 and 1/5 respectively.

# In[ ]:


# 2.
# create a reference to the table from Bike Share dataset
table = client.get_table("bigquery-public-data.austin_bikeshare.bikeshare_trips")

# look at five rows from the dataset to check if we got it
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


table.schema


# In[ ]:


# create a small sample dataset for check
sample_table = client.list_rows(table, max_results=5).to_dataframe()

# get the first cell in the "duration_minutes" column
sample_table.duration_minutes[0]


# In[ ]:


# note for self: never put a comment just before or just after the magic 


# In[ ]:


# 3.


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.bike_model`\nOPTIONS(model_type = 'linear_reg', optimize_strategy = 'batch_gradient_descent') as \nSELECT start_station_name,\n    TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,\n    COUNT(*) as label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n        WHERE start_time < '2018-01-01'\n        GROUP BY start_station_name, start_time             ")


# In[ ]:


# get training statistics from BQML


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM\nML.TRAINING_INFO(MODEL `model_dataset.bike_model`)\nORDER BY iteration')


# In[ ]:


# 4. evaluate the model


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "SELECT * FROM\nML.EVALUATE(MODEL `model_dataset.bike_model`, (\nSELECT start_station_name,\n    TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,\n    COUNT(*) as label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n        WHERE start_time < '2018-01-01'\n        GROUP BY start_station_name, start_time ))")


# In[ ]:


# 5.


# ##Theory for poor performance
# One of the possibilities for poor performance could be the amount of data. Too less data may not train the model enough (which we probably call underfitting).

# In[ ]:


# 6. let's predict something


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "SELECT COUNT(*) AS count_rides FROM\nML.PREDICT(MODEL `model_dataset.bike_model`, (\nSELECT start_station_name,\n    TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,\n    COUNT(*) as label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n        WHERE start_time < '2018-01-01'\n        GROUP BY start_station_name, start_time))\nWHERE start_station_name='22nd & Pearl'")


# In[ ]:


# 7. average daily rides per station


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT EXTRACT(YEAR from start_time) as year,\nCOUNT(DISTINCT(bikeid)) as num_bikes\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nGROUP BY year\nORDER BY year')


# In[ ]:


# 8.


# ##Why this model fails:
# As per results of the exercise, the reason why the ML model failed was due to the spike in ridership in 2018. The data used to train the model has a general downward trend in ridership (data from 2013, but not including 2018).
# 
# Ways to improve ML output:
# * Use data from 2013 to 2018 to account for both the increase and decrease in ridership trends over the years to train the model.
# * Use a polynomial regression model, which is not supported yet in BQML.
# * Treat 2018 data as an outlier and not use it (a dubious approach, considering the substantial data size)
