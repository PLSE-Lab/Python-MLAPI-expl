#!/usr/bin/env python
# coding: utf-8

# # Predict the Spreading of Coronavirus in March 2020
# * A sample submission for the [Coronavirus Task](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/tasks?taskId=508)
# * https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/tasks?taskId=508
# 
# The goal of this task is to build a model that predicts the progression of the virus throughout March 2020.

# Submit a notebook that implements the full lifecycle of data preparation, model creation and evaluation. Feel free to use this dataset plus any other data you have available. Since this is not a formal competition, you're not submitting a single submission file, but rather your whole approach to building a model.
# 
# With this model, you should produce a table in the following format for all future days of March (similar to covid_19_data.csv)
# 
# ```
# ObservationDate: Observation date in mm/dd/yyyy
# Province/State: Province or State
# Country/Region: Country or region
# Confirmed: Cumulative number of confirmed cases
# Deaths: Cumulative number of deaths cases
# Recovered: Cumulative number of recovered cases
# ```
# The notebook should be well documented and contain:
# 
# ```
# Any steps you're taking to prepare the data, including references to external data sources
# Training of your model
# The table mentioned above
# An evaluation of your table against the real data. Let's keep it simple and measure Mean Absolute Error.
# ```

# For my submission I will use only the data that was provided.  My model predicts based off of past data that the number of cases in March will be exactly double the number of cases in January in every province, state, and country.  The results are saved as a .csv file at /kaggle/working/my_submission.csv.

# In[ ]:


# Import Python Packages
import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# Load the Data
nCoV_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
nCoV_data['Date'] = pd.to_datetime(nCoV_data['Date']).dt.normalize()
nCoV_data['ObservationDate'] = nCoV_data.Date.astype(str)
nCoV_data['Country/Region'] = nCoV_data['Country']
nCoV_data = nCoV_data[['ObservationDate','Province/State', 'Country/Region','Confirmed', 'Deaths', 'Recovered']]

# Filter Data for January 2020 Only
january = pd.date_range('2020-01', '2020-02', freq='D')
january = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
               '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12',
               '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16',
               '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20',
               '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',
               '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28',
               '2020-01-29', '2020-01-30', '2020-01-31']
nCoV_data_january_2020 = nCoV_data[nCoV_data['ObservationDate'].isin(january)]

# Double it and swap the date column for dates in March
df = nCoV_data_january_2020
df.Confirmed = df.Confirmed*2
df.Deaths = df.Deaths*2
df.Recovered = df.Recovered*2
nCoV_data_march_2020 = df
nCoV_data_march_2020['ObservationDate'] = nCoV_data_march_2020['ObservationDate'].str.replace("01",'03')
nCoV_data_march_2020.to_csv('/kaggle/working/my_submission.csv',index=False)


# In[ ]:


nCoV_data_march_2020.head(20)


# The next step would be to create an evaluation metric to compare my table against the real data as it comes in.
# 
