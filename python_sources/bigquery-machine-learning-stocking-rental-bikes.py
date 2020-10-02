#!/usr/bin/env python
# coding: utf-8

# # Stocking rental bikes
# 
# ![bike rentals](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg/640px-Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg)
# 
# You stock bikes for a bike rental company in Austin, ensuring stations have enough bikes for all their riders. You decide to build a model to predict how many riders will start from each station during each hour, capturing patterns in seasonality, time of day, day of the week, etc.
# 
# To get started, create a project in GCP and connect to it by running the code cell below. Make sure you have connected the kernel to your GCP account in Settings.

# In[ ]:


# Set your own project id here
PROJECT_ID = 'samueljklee-bqml'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# ## Linear Regression
# 
# Your dataset is quite large. BigQuery is especially efficient with large datasets, so you'll use BigQuery-ML (called BQML) to build your model. BQML uses a "linear regression" model when predicting numeric outcomes, like the number of riders.
# 
# ## 1) Training vs testing
# 
# You'll want to test your model on data it hasn't seen before (for reasons described in the [Intro to Machine Learning Micro-Course](https://www.kaggle.com/learn/intro-to-machine-learning). What do you think is a good approach to splitting the data? What data should we use to train, what data should we use for test the model?

# According to [BigQuery ML data split method](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create):
# 
# > Training data is used to train the model. Evaluation data is used to avoid overfitting via early stopping.
# 
# ```
# When there are fewer than 500 rows in the input data, all rows are used as training data.
# 
# When there are between 500 and 50,000 rows in the input data, 20% of the data is used as evaluation data in a RANDOM split.
# 
# When there are more than 50,000 rows in the input data, only 10,000 of them are used as evaluation data in a RANDOM split.
# ```

# ## Training data
# 
# First, you'll write a query to get the data for model-building. You can use the public Austin bike share dataset from the `bigquery-public-data.austin_bikeshare.bikeshare_trips` table. You predict the number of rides based on the station where the trip starts and the hour when the trip started. Use the `TIMESTAMP_TRUNC` function to truncate the start time to the hour.

# In[ ]:


table = client.get_table("bigquery-public-data.austin_bikeshare.bikeshare_trips")

client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


table.schema


# ## 2) Exercise: Query the training data
# 
# Write the query to retrieve your training data. The fields should be:
# 1. The start_station_name
# 2. A time trips start, to the nearest hour. Get this with `TIMESTAMP_TRUNC(start_time, HOUR) as start_hour`
# 3. The number of rides starting at the station during the hour. Call this `num_rides`.
# Select only the data before 2018-01-01 (so we can save data from 2018 as testing data.)

# In[ ]:


num_rides_query = """
                    SELECT
                        CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,
                        CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,
                        CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,
                        CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,
                        start_station_name,
                        COUNT(IFNULL(bikeid, "1")) AS label
                    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
                    WHERE 
                        DATE(start_time) < '2018-01-01'
                        AND start_station_name IS NOT NULL
                    GROUP BY start_station_name, year, week, day_of_week, hour
                    ORDER BY label DESC
                """

num_rides_job = client.query(num_rides_query).result().to_dataframe()
num_rides_job.head()


# In[ ]:


num_rides_job.label.mean()


# In[ ]:


set(num_rides_job['start_station_name'])


# In[ ]:


num_rides_job.info()


# You'll want to inspect your data to ensure it looks like what you expect. Run the line below to get a quick view of the data, and feel free to explore it more if you'd like (if you don't know how to do that, the [Pandas micro-course](https://www.kaggle.com/learn/pandas)) might be helpful.

# ## Model creation
# 
# Now it's time to turn this data into a model. You'll use the `CREATE MODEL` statement that has a structure like: 
# 
# ```sql
# CREATE OR REPLACE MODEL`model_dataset.bike_trips`
# OPTIONS(model_type='linear_reg') AS 
# -- training data query goes here
# SELECT ...
#     column_with_labels AS label
#     column_with_data_1 
#     column_with_data_2
# FROM ... 
# WHERE ... (Optional)
# GROUP BY ... (Optional)
# ```
# 
# The `model_type` and `optimize_strategy` shown here are good parameters to use in general for predicting numeric outcomes with BQML.
# 
# **Tip:** Using ```CREATE OR REPLACE MODEL``` rather than just ```CREATE MODEL``` ensures you don't get an error if you want to run this command again without first deleting the model you've created.

# ## 3) Exercise: Create and train the model
# 
# Below, write your query to create and train a linear regression model on the training data.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE MODEL `model_dataset.bike_trips`\nOPTIONS(model_type=\'linear_reg\') AS \nSELECT\n    CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,\n    CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,\n    CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,\n    CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,\n    start_station_name,\n    COUNT(IFNULL(bikeid, "1")) AS label\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE \n    DATE(start_time) < \'2018-01-01\'\n    AND start_station_name IS NOT NULL\nGROUP BY start_station_name, year, week, day_of_week, hour')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `model_dataset.bike_trips`)\nORDER BY iteration ')


# ## 4) Exercise: Model evaluation
# 
# Now that you have a model, evaluate it's performance on data from 2018. 
# 
# 
# > Note that the ML.EVALUATE function will return different metrics depending on what's appropriate for your specific model. You can just use the regular ML.EVALUATE funciton here. (ROC curves are generally used to evaluate binary problems, not linear regression, so there's no reason to plot one here.)

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `model_dataset.bike_trips`, (\n    SELECT\n        CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,\n        CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,\n        CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,\n        CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,\n        start_station_name,\n        COUNT(IFNULL(bikeid, "1")) AS label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE \n        DATE(start_time) >= \'2018-01-01\'\n        AND start_station_name IS NOT NULL\n    GROUP BY start_station_name, year, week, day_of_week, hour\n))')


# You should see that the r^2 score here is negative. Negative values indicate that the model is worse than just predicting the mean rides for each example.
# 
# ## 5) Theories for poor performance
# 
# Why would your model be doing worse than making the most simple prediction based on historical data?

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "SELECT\n  start_station_name,\n  COUNT(predicted_label) as total_predicted_rides\nFROM ML.PREDICT(MODEL `model_dataset.bike_trips`, (\n    SELECT\n        CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,\n        CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,\n        CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,\n        CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,\n        start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE \n        DATE(start_time) >= '2018-01-01'\n        AND DATE(start_time) < '2019-01-01'\n        AND start_station_name IS NOT NULL\n    GROUP BY start_station_name, year, week, day_of_week, hour))\n  GROUP BY start_station_name\n  ORDER BY total_predicted_rides DESC\n  LIMIT 10")


# Potential reasons:
# - not enough training data? too wide of a spead of `start_hour` mapped to many `station_id`?
# - data is too "scattered"?
# 
# Tried:
# - given more features (spread out timestamp to day, hour year, day of week)
# 
# Results:
# - better r2 score

# ## 6) Exercise: Looking at predictions
# 
# A good way to figure out where your model is going wrong is to look closer at a small set of predictions. Use your model to predict the number of rides for the 22nd & Pearl station in 2018. Compare the mean values of predicted vs actual riders.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'henderson_prediction', 'SELECT\n  start_station_name,\n  hour,\n  COUNT(predicted_label) as total_predicted_rides,\n  actual_label\nFROM ML.PREDICT(MODEL `model_dataset.bike_trips`, (\n    SELECT\n        CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,\n        CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,\n        CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,\n        CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,\n        start_station_name,\n        COUNT(IFNULL(bikeid, "1")) AS actual_label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE \n        DATE(start_time) >= \'2018-01-01\'\n        AND DATE(start_time) < \'2019-01-01\' \n        AND start_station_name IS NOT NULL\n        AND start_station_name = \'Henderson & 9th\'\n    GROUP BY start_station_name, year, week, day_of_week, hour))\n  GROUP BY start_station_name, hour, actual_label\n  ORDER BY total_predicted_rides DESC\n  LIMIT 50')


# In[ ]:


predicted_mean = henderson_prediction['total_predicted_rides'].mean()
actual_mean = henderson_prediction['actual_label'].mean()
print('Mean value of predicted riders:',predicted_mean)
print('Mean value of actual riders: ',actual_mean)
henderson_prediction.head()


# What you should see here is that the model is underestimating the number of rides by quite a bit. 
# 
# ## 7) Exercise: Average daily rides per station
# 
# Either something is wrong with the model or something surprising is happening in the 2018 data. 
# 
# What could be happening in the data? Write a query to get the average number of riders per station for each year in the dataset and order by the year so you can see the trend. You can use the `EXTRACT` method to get the day and year from the start time timestamp. (You can read up on EXTRACT [in this lesson in the Intro to SQL course](https://www.kaggle.com/dansbecker/order-by)). 

# In[ ]:


num_rides_henderson_query = """
                    SELECT
                        CAST(EXTRACT(YEAR FROM start_time) AS STRING) AS year,
                        CAST(EXTRACT(WEEK FROM start_time) AS STRING) AS week,
                        CAST(EXTRACT(DAYOFWEEK FROM start_time) AS STRING) AS day_of_week,
                        CAST(EXTRACT(HOUR FROM TIMESTAMP_TRUNC(start_time, HOUR)) AS STRING) AS hour,
                        start_station_name,
                        COUNT(IFNULL(bikeid, "1")) AS num_rides
                    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
                    WHERE 
                        DATE(start_time) >= '2018-01-01' 
                        AND DATE(start_time) < '2019-01-01' 
                        AND start_station_name = 'Henderson & 9th'
                    GROUP BY start_station_name, year, week, day_of_week, hour
                    ORDER BY num_rides DESC
                """

num_rides_henderson_job = client.query(num_rides_henderson_query).result().to_dataframe()
num_rides_henderson_job.head()


# ## 8) What do your results tell you?
# 
# Given the daily average riders per station over the years, does it make sense that the model is failing?
# - yes, since the number of riders . over the years isn't consistent and isn't linear. (Shown below)
# 
# ## Next steps
# - Since the count of rides are small, it would be better to use Poisson Regression.
# - Linear regression is more suitable for continuous data
# - Tested with more features (breaking down timestamp), gave better result

# In[ ]:


num_rides_per_year_2017_query = """
                    SELECT
                        EXTRACT(YEAR FROM start_time) AS year,
                        EXTRACT(WEEK FROM start_time) AS week,
                        start_station_name,
                        COUNT(IFNULL(bikeid, "1")) AS num_rides
                    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
                    WHERE 
                        EXTRACT(YEAR FROM start_time) = 2017
                        AND start_station_name = 'Henderson & 9th'
                    GROUP BY start_station_name, year, week
                    ORDER BY week DESC
                """
num_rides_per_year_2017_job = client.query(num_rides_per_year_2017_query).result().to_dataframe()

num_rides_per_year_2018_query = """
                    SELECT
                        EXTRACT(YEAR FROM start_time) AS year,
                        EXTRACT(WEEK FROM start_time) AS week,
                        start_station_name,
                        COUNT(IFNULL(bikeid, "1")) AS num_rides
                    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
                    WHERE 
                        EXTRACT(YEAR FROM start_time) = 2018
                        AND start_station_name = 'Henderson & 9th'
                    GROUP BY start_station_name, year, week
                    ORDER BY week DESC
                """
num_rides_per_year_2018_job = client.query(num_rides_per_year_2018_query).result().to_dataframe()


# In[ ]:


list(zip(num_rides_per_year_2017_job.week,num_rides_per_year_2017_job.num_rides,num_rides_per_year_2018_job.num_rides))


# In[ ]:




