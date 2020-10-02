#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv("/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv")
display(dataset.head())
print("Shape of Dataset : ", dataset.shape)


# Let's checkout the health of these columns. 

# In[ ]:


dataset.dtypes


# In[ ]:


dataset["room_id/id"].nunique()


# 1. Since the unique values of column *room_id/id* is 1, we can drop this column alongwith the *id* column. 
# 2. Also column name 'out/in' isn't a good column name, let's rename it to 'Out_In'.
# 3. We can also see the presence of some duplicate records. We need to handle those as well.

# In[ ]:


dataset.drop(columns=['room_id/id','id'], inplace=True)


# In[ ]:


dataset.rename(columns={'out/in':"Out_In"}, inplace=True)


# In[ ]:


print("Before duplicate treatment : ", dataset.shape)
dataset.drop_duplicates(subset=['noted_date','Out_In','temp'], keep='first',inplace=True)
print("After duplicate treatment : ", dataset.shape)


# Let's check the timespan in which the data was recorded.

# In[ ]:


print("Data recorded from {} to {}".format(min(dataset.noted_date), max(dataset.noted_date)))
print("Minimum Temperature : {} degrees \nMaximum Temperature : {} degrees".format(min(dataset.temp), max(dataset.temp)))


# Good! We have about an year's worth of data recorded from 1st November,2018 to 31st October,2018.
# Let's have a look at the frequency of the sensor in a day i.e. how many times in a day the sensor measures the temperature.
# 
# Let's create the column for the _date_ first, since the *noted_date* has both DD/MM/YYY and the HH:MM components together.
# 
# We can use the _noted_date_ column to create a lot of columns basis date and time! 
# 
# Let's go!

# In[ ]:


dataset['date'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').date())


dataset['day'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').day)
dataset['month'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').month)
dataset['year'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').year)

dataset['hour'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').hour)
dataset['minute'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').minute)
dataset.drop(columns = ['noted_date'], inplace = True)
dataset.head()


# In[ ]:


grouped_by_date = pd.DataFrame(dataset.groupby('date')['minute'].count())
min_freq = grouped_by_date.min()
max_freq = grouped_by_date.max()
grouped_by_date.reset_index(inplace=True)
print(type(grouped_by_date))
display(grouped_by_date.head())


# In[ ]:


grouped_by_date.rename(columns = {"date": "date", 
                                  "minute":"frequency"}, inplace = True)
grouped_by_date.head()
plt.figure(figsize=(5,9))
_ = plt.boxplot(grouped_by_date.frequency)


# Let's see how the Inside and Outside temperature varies across time. Let's also create a Inside-Outside flag to be used later for plotting purposes.

# In[ ]:


def ioflag(x):
    if x=='In':
        return 1
    else:
        return 0
dataset['IO_Flag'] = dataset['Out_In'].apply(lambda x: ioflag(x))
dataset.head()


# In[ ]:


print("Number of records from dataset for Out : ", dataset[dataset['Out_In'] == 'Out'].shape[0]
,"\nNumber of records from dataset for In :  ",dataset[dataset['Out_In'] == 'In'].shape[0])
print("% Distribution for Outside/Inside is 73:28 ")#, 100*(dataset[dataset['Out_In'] == 'Out'].shape[0]/dataset.shape[0]))


# The sensor has recorded Outside temperatures more than the inside. Let's count the number of Outside and Inside temperatures in a day, see if this is a daily behaviour or some dicrepancy.

# In[ ]:


daily_measurement_frequency = pd.DataFrame(dataset.groupby(['date','Out_In'])['temp'].mean())
daily_measurement_frequency.reset_index(inplace=True)
daily_measurement_frequency.rename(columns={'temp':'mean_temperature'}, inplace = True)
daily_measurement_frequency.head()


# In[ ]:


print(type(daily_measurement_frequency))


# In[ ]:


daily_measurement_frequency[daily_measurement_frequency['mean_temperature'] <= 1000 ].head(20)


# ### Mean Temperature 
# Let's plot daily mean temperature basis Inside and Outside

# In[ ]:


colors = {'In':'red', 'Out':'blue'}
plt.figure(figsize=(15,6))
plt.xticks(rotation=70)
plt.scatter(daily_measurement_frequency['date'],
        daily_measurement_frequency['mean_temperature'],
        c=daily_measurement_frequency['Out_In'].apply(lambda x : colors[x]))
plt.show()


# In[ ]:


daily_measurement_variety = pd.DataFrame(dataset[['date','Out_In']].groupby(['date'])['Out_In'].nunique())
daily_measurement_variety.reset_index(inplace=True)
daily_measurement_variety.rename(columns = {'Out_In':'Measurements'}, inplace=True)
daily_measurement_variety.head()


# In[ ]:


plt.figure(figsize=(15,6))
plt.xticks( rotation=70)
plt.yticks(np.arange(0,5,1.0))
plt.scatter(daily_measurement_variety['date'],daily_measurement_variety['Measurements'] )


# In[ ]:


print(daily_measurement_variety['Measurements'].value_counts())


# In[ ]:


dataset.date.nunique()


# In[ ]:


dataset = dataset.merge(daily_measurement_variety, left_on='date', right_on='date')


# In[ ]:


dataset.head()


# In[ ]:


dataset.groupby(['Measurements','Out_In']).count()


# In[ ]:


dataset.groupby(['Measurements','Out_In'])['date'].nunique()


# In[ ]:


dataset.shape


# In[ ]:


monthly_split = pd.DataFrame(dataset.groupby(['month'])['temp'].mean())
monthly_split.reset_index(inplace=True)
monthly_split.rename(columns={'temp':'mean_temp'})
monthly_split.head()


# In[ ]:


display(monthly_split.sort_values(by=['temp'], ascending=False))
display(monthly_split.sort_values(by=['temp'], ascending=True))


# In[ ]:


monthly_type_split = pd.DataFrame(dataset.groupby(['month','Out_In',])['temp'].mean())
monthly_type_split.reset_index(inplace=True)
monthly_type_split.rename(columns={'temp':'mean_temp'})
monthly_type_split.head()


# In[ ]:


display(monthly_type_split[monthly_type_split['Out_In']=='In'].sort_values(by=['temp'], ascending=True))
display(monthly_type_split[monthly_type_split['Out_In']=='Out'].sort_values(by=['temp'], ascending=True))

display(monthly_type_split[monthly_type_split['Out_In']=='In'].sort_values(by=['temp'], ascending=False))
display(monthly_type_split[monthly_type_split['Out_In']=='Out'].sort_values(by=['temp'], ascending=False))


# ---
# ## My Observations so far...
# 
# ### Observations regarding inconsistencies in data
# 1. Even though the max and min date in the dataset give an impression that the sensor carried out measurements for an year, distinct number of days is only 86 days.
# 2. After de-duplication of records, the number of records is 37,268. Out of which, the distribution of outside vs inside temp ratio is 73:27, i.e. 73% of records show a temperature measurement of the 'Outside'.
# 3. Going deeper, we can see that out of 86 days, there were 14 days when only a single type of temperature (either 'In' or 'Out') measurement was taken and during the remaining 72 days both kind of temperature measurement was taken.
# 4. Further, out of the 14 days where only a single type of temperature was measured, 2 days recorded 'Only Inside' temperature and 12 days recorded 'Only Outside' temperature. 
# 5. Out of our 37,368 records, there are 4,148 records where a single type of temperature was recorded and 33,120 records measured both types of temperature.
# 6. Further, out of 4,148 records with single type of temperature, 302 are for 'Inside' and 3,846 are for 'Outside'. Also, out of  33,120 records with both type of temperatures, 9,640 are for 'Inside' and 23,480 are for 'Outside'.
# 
# ### General contextual observations
# 1. Maximum temperature is 51 degrees and minimum temperature is 21 degrees.
# 2. Month with the minimum mean temperature is August and the maximum temperature is October.
# 3. Month with the minimum Inside Temperature is December and maximum Inside Temperature is October.
# 4. Month with the minimum Outside Temperature is August and maximum Outside Temperature is October.
# 

# In[ ]:




