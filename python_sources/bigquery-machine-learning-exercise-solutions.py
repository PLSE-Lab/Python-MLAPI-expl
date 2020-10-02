#!/usr/bin/env python
# coding: utf-8

# ### This exercise is designed to pair with [this tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial). If you haven't taken a look at it yet, head over and check it out first. (Otherwise these exercises will be pretty confusing!) -- Rachael 

# # Stocking rental bikes
# 
# ![bike rentals](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg/640px-Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg)
# 
# You stock bikes for a bike rental company in Austin, ensuring stations have enough bikes for all their riders. You decide to build a model to predict how many riders will start from each station during each hour, capturing patterns in seasonality, time of day, day of the week, etc.
# 
# To get started, create a project in GCP and connect to it by running the code cell below. Make sure you have connected the kernel to your GCP account in Settings.

# In[ ]:


# Set your own project id here
PROJECT_ID = 'kaggle-bqml-briantm' # a string, like 'kaggle-bigquery-240818'

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

# In[ ]:


# You can write your notes here
## Splitting the data base on date, using historical data to predict outcome for later events


# ## Training data
# 
# First, you'll write a query to get the data for model-building. You can use the public Austin bike share dataset from the `bigquery-public-data.austin_bikeshare.bikeshare_trips` table. You predict the number of rides based on the station where the trip starts and the hour when the trip started. Use the `TIMESTAMP_TRUNC` function to truncate the start time to the hour.

# ## 2) Exercise: Query the training data
# 
# Write the query to retrieve your training data. The fields should be:
# 1. The start_station_name
# 2. A time trips start, to the nearest hour. Get this with `TIMESTAMP_TRUNC(start_time, HOUR) as start_hour`
# 3. The number of rides starting at the station during the hour. Call this `num_rides`.
# Select only the data before 2018-01-01 (so we can save data from 2018 as testing data.)

# Write your query below:

# In[ ]:


# create reference to the table
table = client.get_table('bigquery-public-data.austin_bikeshare.bikeshare_trips')

# look at five rows from the table aka. dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'dataframe_name', "SELECT start_station_name, \n        TIMESTAMP_TRUNC(start_time, HOUR) as start_hour, \n        COUNT(*) as num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < '2018-01-01'\nGROUP BY start_station_name, start_hour")


# You'll want to inspect your data to ensure it looks like what you expect. Run the line below to get a quick view of the data, and feel free to explore it more if you'd like (if you don't know how to do that, the [Pandas micro-course](https://www.kaggle.com/learn/pandas)) might be helpful.

# In[ ]:


# look at the dataframe just created
dataframe_name.head()


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
# FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` 
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

# Write your query below:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nCREATE OR REPLACE MODEL `model_dataset.bike_trips`\nOPTIONS(model_type='linear_reg') AS\n\nSELECT start_station_name , \n        TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour, \n        COUNT(*) AS label\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < '2018-01-01'\nGROUP BY start_station_name, start_hour")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', '\nSELECT *\nFROM ML.TRAINING_INFO(MODEL `model_dataset.bike_trips`)\nORDER BY iteration')


# ## 4) Exercise: Model evaluation
# 
# Now that you have a model, evaluate it's performance on data from 2018. 
# 
# 
# > Note that the ML.EVALUATE function will return different metrics depending on what's appropriate for your specific model. You can just use the regular ML.EVALUATE funciton here. (ROC curves are generally used to evaluate binary problems, not linear regression, so there's no reason to plot one here.)

# Write your query below:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nSELECT *\nFROM ML.EVALUATE(MODEL `model_dataset.bike_trips`, (\n    SELECT start_station_name , \n            TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour, \n            COUNT(*) AS label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time >= '2018-01-01'\n    GROUP BY start_station_name, start_hour\n))")


# You should see that the r^2 score here is negative. Negative values indicate that the model is worse than just predicting the mean rides for each example.
# 
# ## 5) Theories for poor performance
# 
# Why would your model be doing worse than making the most simple prediction based on historical data?

# In[ ]:


## Thought question answer here
## Data before 2018 and after 2018 does not correlated


# ## 6) Exercise: Looking at predictions
# 
# A good way to figure out where your model is going wrong is to look closer at a small set of predictions. Use your model to predict the number of rides for the 22nd & Pearl station in 2018. Compare the mean values of predicted vs actual riders.

# Write your query below:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nSELECT AVG(predicted_label) AS predicted_avg_riders, AVG(label) AS actual_avg_riders\nFROM ML.PREDICT(MODEL `model_dataset.bike_trips`, (\n    SELECT  start_station_name , \n            TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour, \n            COUNT(*) AS label\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE   start_time >= '2018-01-01' AND start_time < '2019-01-01' \n            AND start_station_name = '22nd & Pearl'\n    GROUP BY start_station_name, start_hour\n))")


# What you should see here is that the model is underestimating the number of rides by quite a bit. 
# 
# ## 7) Exercise: Average daily rides per station
# 
# Either something is wrong with the model or something surprising is happening in the 2018 data. 
# 
# What could be happening in the data? Write a query to get the average number of riders per station for each year in the dataset and order by the year so you can see the trend. You can use the `EXTRACT` method to get the day and year from the start time timestamp. (You can read up on EXTRACT [in this lesson in the Intro to SQL course](https://www.kaggle.com/dansbecker/order-by)). 

# Write your query below:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', '\nWITH num_daily_rides AS (\n    SELECT COUNT(*) AS num_rides,\n           start_station_name,\n           EXTRACT(DAYOFYEAR from start_time) AS day,\n           EXTRACT(YEAR from start_time) AS year\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    GROUP BY start_station_name, day, year\n    ORDER BY year\n), \nstation_avg AS (\n    SELECT AVG(num_rides) AS avg_riders, \n            start_station_name, \n            year\n    FROM num_daily_rides\n    GROUP BY start_station_name, year\n)\n\nSELECT avg(avg_riders) AS avg_daily_rides_per_station, year\nFROM station_avg\nGROUP BY year\nORDER BY year')


# ## 8) What do your results tell you?
# 
# Given the daily average riders per station over the years, does it make sense that the model is failing?

# In[ ]:


## Thought question answer here
## Yes. The model is falling because number of riders was much more in 2018 than in 2019


# # 9) Next steps
# 
# Given what you've learned, what improvements do you think you could make to your model? Share your ideas on the [Kaggle Learn Forums](https://www.kaggle.com/learn-forum)! (I'll pick a couple of my favorite ideas & send the folks who shared them a Kaggle t-shirt. :)
