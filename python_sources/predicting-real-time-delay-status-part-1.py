#!/usr/bin/env python
# coding: utf-8

# # Section 1: Loading packages and data

# ## 1.1 Downloading and importing packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1.2 Loading dataset

# There are three files: Two csv files and one sql database file.

# First load the trainView.csv into pandas as a dataframe called trainView_df. Display first 5 rows and check if the loading is Okay.

# In[ ]:


file_path = '/kaggle/input/on-time-performance/trainView.csv'
trainView_df = pd.read_csv(file_path)
trainView_df.head()


# Then load the otp.csv into pandas as a dataframe called otp_df. Display first 5 rows and check if the loading is Okay.

# In[ ]:


file_path = '/kaggle/input/on-time-performance/otp.csv'
otp_df = pd.read_csv(file_path)
otp_df.head()


# Last connect to database sqlite file and pull out tables from it.

# In[ ]:


con = sqlite3.connect('/kaggle/input/on-time-performance/database.sqlite') 
tables_df = pd.read_sql_query('SELECT name FROM sqlite_master WHERE type="table"', con)
tables_df


# One table called otp is found from the database sqlite file. Load the otp table into pandas. Display the first 5 rows and check if the loading is Okay.

# In[ ]:


otp_sqlite_df = pd.read_sql_query('SELECT * FROM otp', con)
otp_sqlite_df.head()


# # Section 2: Introduction and data description

# ## 2.1 Introduction

# As one of cummuters using SEPTA train, one of the things I check most frequently in one day is the status of my train.
# 
# However, with a premature prediction system, the delay time predicted by the system is uaually inaccurate. It's extremely normal that the actual arrival time will be 3-5 minutes later than the prediction. And under severe weather conditions, the scenarios could be even worse. I have experienced that the train delayed for more than 45 minutes while the system kept saying the delay time is around 10 minutes.
# 
# Therefore, it's worth implementing big data techniques to develop a machine learning model to give out better real-time prediction.

# ## 2.2 Choose which dataframes to use

# Since there are two dataframes with the same name, otp. The first thing I have done is to figure out if there are any differences between them.

# Without changing any format, first print out the info of otp_df imported from otp.csv file.

# In[ ]:


otp_df.info()


# Then, without changing any format, print out the info of otp_sqlite_df imported from data.sqlite file.

# In[ ]:


otp_sqlite_df.info()


# The above printouts clearly showed that both dataframes have same number of columns with exactly the same names.

# Further verify differences: Print out the describe into of each dataframe and compare.

# In[ ]:


print(otp_df.describe())
print(otp_sqlite_df.describe())


# The comparison above showed that the information contained by two dataframes are identical. Therefore, using either one for analysis should work. I will use otp.df for all following analysis

# ## 2.3 Data description and getting data into right format

# There are two dataframes in total that will be used for my final project. One is the otp_df, and the other is trainView_df.

# ### 2.3.1 On-time performace dataframe (otp_df)

# First, let's look into otp_df

# In[ ]:


otp_df.info()


# According to the above results, otp_df has 1,882,015 instances and each comes with 7 columns.
# 
# 1.   train_id
# 2.   direction ('N' or 'S' direction is demarcated as either Northbound or Southbound)
# 3.   origin
# 4.   next_station (station stop, at timeStamp)
# 1.   date
# 2.   status ('On Time', '5 min', '10 min'. This is a status on train lateness. 999 is a suspended train)
# 3.   timeStamp
# 

# Convert all datetime data into correct format.

# In[ ]:


otp_df['date'] = pd.to_datetime(otp_df['date'])
otp_df['timeStamp'] = pd.to_datetime(otp_df['timeStamp'])


# Since all delay time is in minutes, drop seconds term in timestamp.

# In[ ]:


otp_df['timeStamp'] = otp_df['timeStamp'].apply(lambda x: x.replace(second=0))


# Create a self-defined function to convert current status to delay time in minutes (int type).

# In[ ]:


def status2min(x):
  x_ls = x.split(' ')
  if x == 'On Time':
    return 0
  elif x_ls[1] == 'min':
    return int(x_ls[0])
  else:
    raise ValueError

otp_df['status'] = otp_df['status'].apply(status2min)


# Create a new column called "status_dt" to store delay time in timedelta format.

# In[ ]:


otp_df['status_dt'] = otp_df['status'].apply(lambda x: datetime.timedelta(minutes=x))


# Print out the info and first 5 rows for final check.

# In[ ]:


otp_df.info()


# In[ ]:


otp_df.head()


# ### 2.3.2 GPS train dataframe (trainView_df)

# First, take a look at trainView_df.

# In[ ]:


trainView_df.info()


# According to the above results, trainView_df has 3,601,656 instances and each comes with 14 columns.
# 
# 1.   train_id
# 2.   status
# 4.   next_station
# 1.   service
# 2.   dest
# 1.   lon
# 2.   lat
# 3.   source
# 1.   track_change
# 2.   track
# 1.   date
# 1.   timeStamp0 (first timeStamp at coordinates)
# 2.   timeStamp1 (last timeStamp at coordinates)
# 2.   secondes (duration at coordinates)

# Convert all datetime data into correct format.

# In[ ]:


trainView_df['date'] = pd.to_datetime(trainView_df['date'])
trainView_df['timeStamp0'] = pd.to_datetime(trainView_df['timeStamp0'])
trainView_df['timeStamp1'] = pd.to_datetime(trainView_df['timeStamp1'])


# Since all delay time is in minutes, drop seconds term in both timestamp.

# In[ ]:


trainView_df['timeStamp0'] = trainView_df['timeStamp0'].apply(lambda x: x.replace(second=0))
trainView_df['timeStamp1'] = trainView_df['timeStamp1'].apply(lambda x: x.replace(second=0))


# Create a self-defined function to convert current status to delay time in minutes (int type).

# In[ ]:


def status2int(x):
  if x == 'None':
    return 0
  else:
    return int(x)

trainView_df['status'] = trainView_df['status'].apply(status2int)


# Create a new column called "status_dt" to store delay time in timedelta format.

# In[ ]:


trainView_df['status_dt'] = trainView_df['status'].apply(lambda x: datetime.timedelta(minutes=x))


# Print out the info and first 5 rows for final check.

# In[ ]:


trainView_df.info()


# In[ ]:


trainView_df.head()


# # Section 3: Exploratory Data Analysis

# ## 3.1 Is train_id unique?

# For example, randomly pick up one train_id of "598" from otp_df and plot how many trains with "598" as train_id running every day.

# In[ ]:


test_df = otp_df[otp_df['train_id'] == '598'].copy()
test_df = test_df.groupby('date')['train_id'].count().reset_index()

fig, ax = plt.subplots(figsize=[10,5])
ax.bar(test_df['date'], test_df['train_id'])
ax.set_xlabel('Date')
ax.set_ylabel('Show-up times')
ax.set_title('How many times the train id 598 showed up in each day?')
plt.show()


# The plot above shows that the train_id is not unique. It clearly shows that the same train_id can show up in multiple days.

# Plot how many trains with "598" as train_id running each hour on 2016-03-24 from otp_df.

# In[ ]:


test_df = otp_df[(otp_df['date'] == '2016-03-24') & (otp_df['train_id'] == '598')].copy()
test_df['hour'] = test_df['timeStamp'].dt.hour
test_df = test_df.groupby('hour')['train_id'].count().reset_index()

fig, ax = plt.subplots(figsize=[10,5])
ax.bar(test_df['hour'], test_df['train_id'])
ax.set_xlabel('Hour')
ax.set_ylabel('Show-up times')
ax.set_title('How many times the train id 598 showed up in each hour?')
plt.show()


# Since the longest run for a SEPTA train is less than 2 hours, the plot above shows that there are unless two trains using the same train_id running in a single day. Therefore, the train_id cannot be used as the unique key to identify each train in the system.

# ## 3.2 Does the number of train running in the system related to the Month/Day of week/Hour?

# Plot the total number of trains running each day using data from trainView_df.

# In[ ]:


test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])
test_df = test_df.groupby('date')['train_id'].count()

fig, ax = plt.subplots(figsize=[10,5])
ax.plot(test_df)
ax.set_xlabel('Date')
ax.set_ylabel('Num of trains')
ax.set_title('How many trains are running every day?')
plt.show()


# Clearly, the above plot shows that the total number of trains running each day depends on Month and Day of week.

# To be more specific, plot average train numbers running every day per month.

# In[ ]:


test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])
test_df = test_df.groupby('date')['train_id'].count().reset_index()
test_df['month'] = test_df['date'].dt.month
test_df = test_df[['month', 'train_id']].groupby('month').agg(np.mean).reset_index()

fig, ax = plt.subplots(figsize=[7,5])
ax.bar(test_df['month'], test_df['train_id'])
ax.set_xlabel('Month')
ax.set_ylabel('Number of trains')
ax.set_title('Average train numbers running every day in each month')
plt.show()


# To be more specific, plot average train numbers running each day of week.

# In[ ]:


test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])
test_df = test_df.groupby('date')['train_id'].count().reset_index()
test_df['DOW'] = test_df['date'].dt.dayofweek + 1
test_df = test_df.groupby('DOW')['train_id'].agg(np.mean).reset_index()
test_df

fig, ax = plt.subplots(figsize=[7,5])
sns.barplot(x='DOW', y='train_id', data=test_df, palette="Blues_d")
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average number of trains')
ax.set_title('Average train numbers running each day of week')
plt.show()


# Because the scale of the first plot is large, in order to know if there is any variance of train number across each hour, I will plot average train numbers each hour.

# In[ ]:


test_df = trainView_df.copy()
test_df['hour'] = test_df['timeStamp0'].dt.hour
test_df.drop_duplicates(subset=['train_id', 'date', 'hour'], inplace=True)
test_df = test_df.groupby(['date','hour'])['train_id'].count().reset_index()
test_df = test_df.groupby('hour').agg(np.mean).reset_index()

fig, ax = plt.subplots(figsize=[10,5])
sns.barplot(x='hour', y='train_id', data=test_df, palette="Blues_d")
ax.set_xlabel('Hour')
ax.set_ylabel('Average number of trains')
ax.set_title('Average train numbers running each hour')
plt.show()


# From all plots above, the train numbers have strong relationship with Month/Day of week/Hour. Therefore, all of these will considered as features for later machine learning models.

# ## 3.3 What's important in trainView_df

# What does "service" column have?

# In[ ]:


test_df = trainView_df.copy()
test_df['service'] = test_df['service'].str.upper()
test_df = test_df.groupby('service')['train_id'].count()
test_df = test_df.sort_values(ascending=False).head(20).reset_index()

fig, ax = plt.subplots(figsize=[5,8])
sns.barplot(y='service', x='train_id', data=test_df, palette="Blues_d")
ax.set_xlabel('Count (log scale)')
ax.set_ylabel('Service type')
ax.set_title('Top 10 frequent service type')
ax.set_xscale('log')
plt.show()

# release the RAM storing test_df
test_df = []


# The "service" column labels trains with service type. Based on the list of service column printed out above, generally there are two types of service: Local and Express.
# 
# For each servie line, both Local and Express trains are running on the same route. However, Local trains stop at each train station while Express trains skip the stops where fewer amount of people are using.
# 
# I decide to drop this column since the service type also reflects in "next_station" column.

# Clean trainView_df by dropping unnecessary columns.

# In[ ]:


columns_to_drop = ['lon', 'lat', 'track_change', 'track',
                   'service', 'timeStamp1', 'seconds']

trainView_df = trainView_df.drop(columns=columns_to_drop)


# Trains suspended are labeled with 999 in the status column. Since suspension is usually due to mechanical problems, I will no longer consider trains suspended.

# In[ ]:


trainView_df = trainView_df[trainView_df['status'] < 999]


# Have a closer look at the distribution of dalay time in trainView_df and plot it out.

# In[ ]:


trainView_df['status'].describe()


# In[ ]:


fig, ax = plt.subplots(figsize=[10,5])
sns.distplot(trainView_df['status'])
ax.set_xlabel('Delay time (min)')
ax.set_ylabel('Density')
ax.set_title('Distribution of delay time')
plt.show()


# ## 3.4 Build train schedule based on otp_df

# The idea here is that I want to feed the machine learn algorithm with as much information as I have. However, when giving out prediction on a real-time manner, one instance should not see any information that happened after it. Therefore, I create a schedule dataframe and assume that all instances have access to it and know how crowded is the overall system as well as how crowded is each specific train station.

# First drop instances where next_station is None and the trains are suspended.

# In[ ]:


otp_df = otp_df[otp_df['next_station'] != 'None']
otp_df = otp_df[otp_df['status'] < 999]


# Based on otp_df, create a schedule dataframe called schedule_df:
# 
# * Initialize 
# * Change name of columns accordingly
# * Create arrival_time column using (timeStamp - delay time)
# * Extract date and hour from arrival_time
# * Keep necessary columns
# * Print out first 5 rows and check

# In[ ]:


schedule_df = otp_df.copy()

change_col_name = {'next_station': 'arrival_station'}
schedule_df = schedule_df.rename(columns=change_col_name)

schedule_df['arrival_time'] = schedule_df['timeStamp'] - schedule_df['status_dt']
schedule_df['arrival_hour'] = schedule_df['arrival_time'].dt.hour
schedule_df['arrival_date'] = schedule_df['arrival_time'].apply(lambda x: x.date())

col_to_keep = ['train_id', 'direction', 'origin', 'arrival_station', 'date',
               'arrival_date', 'arrival_hour']
schedule_df = schedule_df[col_to_keep]

schedule_df.head()


# Verify the schedule by picking up a random example.

# In[ ]:


schedule_df[schedule_df['train_id'] == '778'].head(10)


# # Section 4: Build feature and target matrix

# ## 4.1 Prepare feature matrix

# ### 4.1.1 Add direction to trainView_df by joining with the direction_df pull out from otp_df

# Since direction in trainView_df is missing, first join directions onto trainView_df. Since the train_id is not unique, need more keys to make sure that mismatch will not happen.
# 
# keys:
# * train_id
# * date
# * hour
# * next_station

# Prepare keys: hour.

# In[ ]:


otp_df['hour'] = otp_df['timeStamp'].dt.hour


# In[ ]:


trainView_df['hour'] = trainView_df['timeStamp0'].dt.hour


# Create otp_direction_label_df which has all 4 keys plus direction lable. Print out first 5 rows and check.

# In[ ]:


otp_direction_label_df = otp_df.copy()
col_to_keep = ['train_id', 'direction', 'next_station', 'date', 'hour']
otp_direction_label_df = otp_direction_label_df[col_to_keep]
otp_direction_label_df = otp_direction_label_df.drop_duplicates()
otp_direction_label_df.head()


# Merge trainView_df with direction label and save as trainView_dir_df. Print out first 5 rows and check.

# In[ ]:


trainView_dir_df = trainView_df.merge(otp_direction_label_df, indicator=True, 
                                      left_on=['train_id', 'next_station', 'date', 'hour'],
                                      right_on=['train_id', 'next_station', 'date', 'hour'])
trainView_dir_df.head()


# Compare the shape before and after merging. The size after merging is smaller and it makes sense.

# In[ ]:


print(trainView_df.shape)
print(trainView_dir_df.shape)


# ### 4.1.2 Combine two dataframes

# First prepare trainView_dir_df. Only keep the necessary columns for fianl analysis. Drop duplicates, and print out the shape and first 5 rows.

# In[ ]:


col_to_keep = ['train_id', 'direction', 'status', 'next_station', 
               'date', 'timeStamp0', 'hour']
trainView_dir_df = trainView_dir_df[col_to_keep]
trainView_dir_df = trainView_dir_df.drop_duplicates()
print(trainView_dir_df.shape)
trainView_dir_df.head()


# Correct column names and make sure the names from two dataframes match. Print out the first 5 rows.

# In[ ]:


change_col_name = {'timeStamp0': 'timeStamp'}
trainView_dir_df = trainView_dir_df.rename(columns=change_col_name)
trainView_dir_df.head()


# Prepare otp dataframe by selecting necessary columns and dropping duplicates. Print out first 5 rows and check.

# In[ ]:


otp_combine_df = otp_df.copy()
col_to_keep = ['train_id', 'direction', 'status', 'next_station', 
               'date', 'timeStamp', 'hour']
otp_combine_df = otp_combine_df[col_to_keep]
otp_combine_df = otp_combine_df.drop_duplicates()
otp_combine_df.head()


# Combine two dataframes vertically and drop duplicates. Print out shapes before an after, as well as first 5 rows.

# In[ ]:


combine_df = pd.concat([trainView_dir_df, otp_combine_df]).sort_values('timeStamp')
print(combine_df.shape)
combine_df = combine_df.drop_duplicates()
print(combine_df.shape)
combine_df.head()


# ### 4.1.3 Features based on schedule

# For each station, each direction, and each specific hour, calculate how many trains arrive/leave.

# In[ ]:


station_hour_count_df = schedule_df[['arrival_station', 'direction', 'arrival_date', 'arrival_hour', 'train_id']].copy()
station_hour_count_df = station_hour_count_df.drop_duplicates()
station_hour_count_df = station_hour_count_df.groupby(['arrival_station', 'direction', 'arrival_date', 'arrival_hour']).count().reset_index()
station_hour_count_df = station_hour_count_df.rename(columns={'train_id': 'num_train'})
station_hour_count_df.head()


# For each station, each direction, and each day, calculate how many trains arrive/leave.

# In[ ]:


station_day_count_df = schedule_df[['arrival_station', 'direction', 'arrival_date', 'train_id']].copy()
station_day_count_df = station_day_count_df.drop_duplicates()
station_day_count_df = station_day_count_df.groupby(['arrival_station', 'direction', 'arrival_date']).count().reset_index()
station_day_count_df = station_day_count_df.rename(columns={'train_id': 'num_train'})
station_day_count_df.head()


# For each day, calculate how many trains in total are running in the system.

# In[ ]:


sys_day_count_df = schedule_df[['direction', 'arrival_date', 'train_id']].copy()
sys_day_count_df = sys_day_count_df.drop_duplicates()
sys_day_count_df = sys_day_count_df.groupby(['arrival_date'])['train_id'].count().reset_index()
sys_day_count_df = sys_day_count_df.rename(columns={'train_id': 'num_train'})
sys_day_count_df.head()


# For each specific hour, calculate how many trains in total are running in the system.

# In[ ]:


sys_hour_count_df = schedule_df[['direction', 'arrival_date', 'train_id', 'arrival_hour']].copy()
sys_hour_count_df = sys_hour_count_df.drop_duplicates()
sys_hour_count_df = sys_hour_count_df.groupby(['arrival_date', 'arrival_hour'])['train_id'].count().reset_index()
sys_hour_count_df = sys_hour_count_df.rename(columns={'train_id': 'num_train'})
sys_hour_count_df.head()


# ### 4.1.4 Last delay status of the same train

# Sort combine_df based on time order, use reset_index function to create unique time_sequence id.

# In[ ]:


combine_df = combine_df.sort_values('timeStamp')
combine_df = combine_df.reset_index().rename(columns={'index': 'orig_index'})
combine_df = combine_df.reset_index().rename(columns={'index': 'time_sequence'})
combine_df.head()


# Create a sliding window with width of 2, take the lastest timestamp, create a new column with values shifted by 1. Print out first 5 rows and check.

# In[ ]:


last_df = combine_df[['train_id', 'direction', 'time_sequence']]
last_df['last_time_sequence'] = last_df['time_sequence']
last_df = last_df.groupby(['train_id', 'direction', 'time_sequence']).sum().rolling(2).min().reset_index()
last_df = last_df[last_df['last_time_sequence'] != last_df['time_sequence']].dropna()[['time_sequence', 'last_time_sequence']]
last_df.head()


# Merge with combine_df to get last delay info. Keep useful columns and print first 5 rows.

# In[ ]:


last_df = last_df.merge(combine_df, how='left', left_on='last_time_sequence',
                        right_on='time_sequence')
col_to_keep = ['time_sequence_x', 'train_id', 'direction', 'status', 
               'next_station', 'timeStamp', 'hour']
last_df = last_df[col_to_keep]
last_df.head()


# Rename columns accordingly and drop all rows with NaN. Print out first 5 rows.

# In[ ]:


change_col_name = {'time_sequence_x': 'time_sequence', 'train_id': 'last_train_id', 
                   'direction': 'last_direction', 'status': 'last_status', 
                   'next_station': 'last_next_station', 'timeStamp': 'last_timeStamp',
                   'hour':'last_hour'}
last_df = last_df.rename(columns=change_col_name)
last_df = last_df.dropna()
last_df.head()


# Merge last status with feature matrix. Sort by time stamp and print out first 5 rows.

# In[ ]:


combine_last_df = combine_df.merge(last_df, on=['time_sequence']).sort_values('timeStamp')
combine_last_df.head()


# Drop unnecessary columns and print out first 5 rows.

# In[ ]:


col_to_drop = ['last_train_id', 'last_direction', 'last_next_station', 'last_hour']
col = []
for i in combine_last_df.columns:
  if i not in col_to_drop:
    col.append(i)

combine_last_df = combine_last_df[col]
combine_last_df.head()


# Caculate time difference between the last update. Sort by time difference in descending order. Print out first 5 rows.

# In[ ]:


combine_last_df['delta_T'] = combine_last_df['timeStamp'] - combine_last_df['last_timeStamp']
combine_last_df.sort_values('delta_T', ascending=False).head()


# Plot the distribution of time difference. It doesn't make sense there are time differences greater than 2 hours (7200 s).

# In[ ]:


fig, ax = plt.subplots(figsize=[5, 5])
sns.distplot(combine_last_df['delta_T'].dt.total_seconds())
ax.set_xlabel('Time difference (sec)')
ax.set_ylabel('Density')
ax.set_title('Distribution of time difference')
plt.show()


# That's because mismatch was introduced when I shift the time sequence. Drop rows with time difference greater than 90 minutes. Print out first 5 rows.

# In[ ]:


combine_last_df = combine_last_df[combine_last_df['delta_T'] < datetime.timedelta(minutes=90)]
combine_last_df.head()


# Replot time difference distribution and it makes sense.

# In[ ]:


fig, ax = plt.subplots(figsize=[5, 5])
sns.distplot(combine_last_df['delta_T'].dt.total_seconds())
ax.set_xlabel('Time difference (sec)')
ax.set_ylabel('Density')
ax.set_title('Distribution of time difference')
plt.show()


# Drop unnecessary columns.

# In[ ]:


col_to_drop = ['last_timeStamp']
col = []
for i in combine_last_df.columns:
  if i not in col_to_drop:
    col.append(i)

combine_last_df = combine_last_df[col]
combine_last_df.head()


# Create a column with time difference in minutes as int.

# In[ ]:


combine_last_df['delta_T_int'] = combine_last_df['delta_T'].dt.total_seconds().astype(int)/60


# Finally, verify the results.

# In[ ]:


combine_last_df[combine_last_df['train_id'] == '778']


# In[ ]:


combine_last_df[combine_last_df['date'] == '2016-10-01']


# ### 4.1.5 The average delay time of the system/each train station in the last hour

# Calculate the average delay time in each hour at each station, date, and train_id.

# In[ ]:


avg_delay_df = combine_last_df.copy()
avg_delay_df = avg_delay_df.drop(columns=['time_sequence', 'delta_T', 'delta_T_int', 'last_status', 'timeStamp', 'orig_index'])
station_delay_df = avg_delay_df.groupby(['direction', 'next_station', 'date', 'hour', 'train_id'])['status'].agg(['mean']).reset_index()
station_delay_df.head()


# Further calculate the average delay time in each hour at each station and date.

# In[ ]:


station_delay_df = station_delay_df.groupby(['direction', 'next_station', 'date', 'hour'])['mean'].agg(['mean']).reset_index()
station_delay_df.head()


# Create time stamp by combine date and hour.

# In[ ]:


def combineTime(x):
  y = datetime.datetime.combine(x[0].date(), datetime.time(x[1],0))
  return y

station_delay_df['timeStamp'] = station_delay_df[['date', 'hour']].apply(lambda x: combineTime(x), axis=1)
station_delay_df.head()


# Rename columns accordingly and select necessary info.

# In[ ]:


station_delay_df = station_delay_df.rename(columns={'mean': 'avg_delay'})
station_delay_df = station_delay_df[['direction', 'next_station', 'timeStamp', 'avg_delay']]
station_delay_df.head()


# Calculate the average delay time of each train, each date and hour.

# In[ ]:


sys_delay_df = avg_delay_df.groupby(['date', 'hour', 'train_id'])['status'].agg(['mean']).reset_index()
sys_delay_df.head()


# Further calculate the average delay time of the system in each specific hour.

# In[ ]:


sys_delay_df = sys_delay_df.groupby(['date', 'hour'])['mean'].agg(['mean']).reset_index()
sys_delay_df.head()


# Create time stamp by combining date and time.

# In[ ]:


def combineTime(x):
  y = datetime.datetime.combine(x[0].date(), datetime.time(x[1],0))
  return y

sys_delay_df['timeStamp'] = sys_delay_df[['date', 'hour']].apply(lambda x: combineTime(x), axis=1)
sys_delay_df.head()


# Rename columns and select necessary infomation.

# In[ ]:


sys_delay_df = sys_delay_df.rename(columns={'mean': 'avg_delay'})
sys_delay_df = sys_delay_df[['timeStamp', 'avg_delay']]
sys_delay_df.head()


# ## 4.2 Build feature matrix

# ### 4.2.1 Initialize feature matrix

# Initialize feature matrix based on combine_df, drop unnecessary columns, and change column names accordingly. Print out the inital shape of feature matrix and first 5 rows.

# In[ ]:


feature_df = combine_last_df.copy()
feature_df = feature_df.drop(columns=['time_sequence', 'orig_index', 'delta_T'])
feature_df = feature_df.rename(columns={'delta_T_int': 'delta_T'})
print(feature_df.shape)
feature_df.head()


# ### 4.2.2 Merge with station_day_count_df: same direction

# Prepare station_day_count_df by modifying the name and data type accordingly.

# In[ ]:


station_day_count_df = station_day_count_df.rename(columns={'num_train': 'num_station_day',
                                                            'arrival_station': 'next_station',
                                                            'arrival_date': 'date'})
station_day_count_df['date'] = pd.to_datetime(station_day_count_df['date'])
station_day_count_df.head()


# Perform merge and inspect the shape of feature matrix and first 5 rows.

# In[ ]:


feature_df = feature_df.merge(station_day_count_df, on=['next_station', 'direction', 'date'])
print(feature_df.shape)
feature_df.head()


# Change the column names accordingly.

# In[ ]:


feature_df = feature_df.rename(columns={'num_station_day': 'num_station_day_same'})
feature_df.head()


# ### 4.2.3 Merge with station_day_count_df: opposite direction

# Create a opp_dir column in feature matrix.

# In[ ]:


def op_dir(x):
  if x == 'N':
    return 'S'
  else:
    return 'N'

feature_df['opp_dir'] = feature_df['direction'].apply(op_dir)
feature_df.head()


# Perform merge on three key columns. Inspect the feature matrix shape after merge.

# In[ ]:


left_key = ['next_station', 'opp_dir', 'date']
right_key = ['next_station', 'direction', 'date']
feature_df = feature_df.merge(station_day_count_df, left_on=left_key, right_on=right_key)
feature_df = feature_df.drop(columns='direction_y')
feature_df = feature_df.rename(columns={'num_station_day': 'num_station_day_opp',
                                        'direction_x': 'direction'})
print(feature_df.shape)
feature_df.head()


# ### 4.2.4 Merge with station_hour_count_df: same direction

# Prepare station_hour_count_df by modifying column names and types.

# In[ ]:


change_col_name = {'arrival_station': 'next_station',
                   'arrival_date': 'date',
                   'arrival_hour': 'hour',
                   'num_train': 'num_station_hour'}
station_hour_count_df = station_hour_count_df.rename(columns=change_col_name)

station_hour_count_df['date'] = pd.to_datetime(station_hour_count_df['date'])

station_hour_count_df.head()


# Perform merge on 4 key columns and inspect the shape of feature matrix.

# In[ ]:


feature_df = feature_df.merge(station_hour_count_df, on=['next_station', 'direction', 'date', 'hour'])
print(feature_df.shape)
feature_df.head()


# Rename the columns accordingly.

# In[ ]:


feature_df = feature_df.rename(columns={'num_station_hour': 'num_station_hour_same'})
feature_df.head()


# ### 4.2.5 Merge with station_hour_count_df: opposite direction

# Perform merge on 4 key columns and inspect the shape of feature matrix.

# In[ ]:


left_key = ['next_station', 'opp_dir', 'date', 'hour']
right_key = ['next_station', 'direction', 'date', 'hour']
feature_df = feature_df.merge(station_hour_count_df, left_on=left_key, right_on=right_key)
print(feature_df.shape)
feature_df.head()


# Drop unnecessary columns and rename columns.

# In[ ]:


feature_df = feature_df.drop(columns='direction_y')
feature_df = feature_df.rename(columns={'direction_x': 'direction'})
feature_df = feature_df.rename(columns={'num_station_hour': 'num_station_hour_opp'})
feature_df.head()


# ### 4.2.6 Merge with sys_day_count_df

# Prepare sys_day_count_df by modifying column names and type.

# In[ ]:


sys_day_count_df = sys_day_count_df.rename(columns={'arrival_date': 'date',
                                                    'num_train': 'num_sys_day'})
sys_day_count_df['date'] = pd.to_datetime(sys_day_count_df['date'])
print(sys_day_count_df.shape)
print(sys_day_count_df.drop_duplicates().shape)
sys_day_count_df.head()


# Perform merge and inspect shape of feature matrix.

# In[ ]:


feature_df = feature_df.merge(sys_day_count_df, on=['date'])
print(feature_df.shape)
feature_df.head()


# ### 4.2.7 Merge with sys_hour_count_df

# Prepare sys_hour_count_df by modifying column names and data type.

# In[ ]:


change_col_name = {'arrival_date': 'date',
                   'arrival_hour': 'hour',
                   'num_train': 'num_sys_hour'}
sys_hour_count_df = sys_hour_count_df.rename(columns=change_col_name)

sys_hour_count_df['date'] = pd.to_datetime(sys_hour_count_df['date'])

sys_hour_count_df.head()


# Merge two dataframes on two keys. Inspect shape of feature matrix.

# In[ ]:


feature_df = feature_df.merge(sys_hour_count_df, on=['date', 'hour'])
print(feature_df.shape)
feature_df.head()


# ### 4.2.8 Merge with station_delay_df: same direction

# Prepare station_delay_df by modifing column names.

# In[ ]:


change_col_name = {'timeStamp': 'last_hour',
                   'avg_delay': 'avg_station_same'}
station_delay_df = station_delay_df.rename(columns=change_col_name)
station_delay_df.head()


# Create last_hour column in feature matrix by subtracting 1 hour from current timestamp.

# In[ ]:


def last_hour(x):
  pure_hour = x.replace(minute=0)
  last_hour = pure_hour - datetime.timedelta(hours=1)
  return last_hour

feature_df['last_hour'] = feature_df['timeStamp'].apply(last_hour)
feature_df.head()


# Merge on 3 keys and inspect matrix shape.

# In[ ]:


feature_df = feature_df.merge(station_delay_df, on=['direction', 'next_station',
                                                    'last_hour'])
print(feature_df.shape)
feature_df.head()


# ### 4.2.9 Merge with station_delay_df: opposite direction

# Prepare station_delay_df by modifying column names.

# In[ ]:


change_col_name = {'avg_station_same': 'avg_station_opp'}
station_delay_df = station_delay_df.rename(columns=change_col_name)
station_delay_df.head()


# Merge on 3 keys and inspect matrix shape.

# In[ ]:


left_key = ['opp_dir', 'next_station', 'last_hour']
right_key = ['direction', 'next_station', 'last_hour']
feature_df = feature_df.merge(station_delay_df, left_on=left_key, right_on=right_key)
print(feature_df.shape)
feature_df.head()


# Drop unnecessary columns and change column names accordingly.

# In[ ]:


feature_df = feature_df.drop(columns=['direction_y'])
feature_df = feature_df.rename(columns={'direction_x': 'direction'})
feature_df.head()


# ### 4.2.10 Merge with sys_delay_df

# Prepare sys_delay_df by modifying column names.

# In[ ]:


change_col_name = {'timeStamp': 'last_hour',
                   'avg_delay': 'avg_sys'}
sys_delay_df = sys_delay_df.rename(columns=change_col_name)
sys_delay_df.head()


# Merge on last_hour and inspect the shape of feature matrix.

# In[ ]:


feature_df = feature_df.merge(sys_delay_df, on='last_hour')
print(feature_df.shape)
feature_df.head()


# ### 4.2.11 Get day of week & month of year

# Extract day of week and month from timestamp

# In[ ]:


feature_df['dow'] = feature_df['date'].dt.dayofweek + 1
feature_df['month'] = feature_df['date'].dt.month
feature_df.head()


# ### 4.2.12 Clean feature matrix and get dummies

# Drop all unnecessary columns.

# In[ ]:


col_to_drop = ['train_id', 'date', 'timeStamp', 'opp_dir', 'last_hour', 'next_station']
feature_df = feature_df.drop(columns=col_to_drop)
feature_df.head()


# Get dummies for all categorical columns

# In[ ]:


X = pd.get_dummies(feature_df, columns=['dow', 'month', 'hour', 'direction']).copy()
X = X.drop(columns='status')
print(X.shape)
X.head()


# Create target dataframe.

# In[ ]:


y = feature_df['status'].copy()
y.head()


# Save outputs

# In[ ]:


X.to_csv('X.csv')
y.to_csv('y.csv')

