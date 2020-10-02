#!/usr/bin/env python
# coding: utf-8

# # Required Tasks
# 
# ## Visualize US Accidents Dataset
#     This is a countrywide traffic accident dataset, which covers 49 states of the United States. The purpose of this task is to visualize the dataset states wise.
# 
# ## The state that has the highest number of accidents
#     Task Details
#     Which US state has the highest number of accidents, and a description of the accidents that usually occur in that state.
#     Expected Submission
#     A notebook that shows the state with the highest number of accidents and the main cause of the accidents in that state.
#     Evaluation
#     All states should be ranked with the first one being the state with the highest accidents. This is done for better comparisons and to provide a more detailed solution.
# 
# ## At what time do accidents usually occur in the US
#     Task Details
#     Figure out the time that accidents usually occur in the US. This can be done by using the "start time" and "end time" columns.
#     Expected Submission
#     A notebook that outputs the time that accidents usually occur.
#     Evaluation
#     A proper ranking of the times that accidents occur. The time that accidents usually occur should be at the top of the list together with the number of times accidents occurred during that specific time. Other times at lower ranks should follow. This is done to provide a more detailed solution.
# 
# ## Predict the location of the accident
# 
# ## Factors Affecting Accident Severity
#     Examine the relationship between accident severity and other accident information such as time, weather, and location.

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
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sb


# In[ ]:


df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df.head()


# In[ ]:


df.dtypes


# In[ ]:


# Obtaining all the possible numeric columns
df.describe()


# In[ ]:


#We can make the start and End Times real datetime columns
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
df.dtypes


# In[ ]:


#Lets identify columns with Nulls and NaNs.

df.isnull().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


#Cleanup some columns by removing Nulls or Filling NaNs
#Define a funcation impute median
def impute_median(series):
    return series.fillna(series.median())

def impute_mean(series):
    return series.fillna(series.mean())


# In[ ]:


df.End_Lat = df['End_Lat'].transform(impute_median)
df.End_Lng = df['End_Lng'].transform(impute_median)
df.TMC = df['TMC'].transform(impute_median)
df.Number = df['Number'].transform(impute_mean)
df['Temperature(F)'] = df['Temperature(F)'].transform(impute_median)
df['Wind_Chill(F)'] = df['Wind_Chill(F)'].transform(impute_median)
df['Humidity(%)'] = df['Humidity(%)'].transform(impute_median)
df['Pressure(in)'] = df['Pressure(in)'].transform(impute_median)
df['Visibility(mi)'] = df['Visibility(mi)'].transform(impute_median)
df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].transform(impute_median)
df['Precipitation(in)'] = df['Precipitation(in)'].transform(impute_median)
df.isnull().sum()


# In[ ]:


#Modes of categorical values
print(df['Weather_Condition'].mode())
print(df['Astronomical_Twilight'].mode())
print(df['Nautical_Twilight'].mode())
print(df['Weather_Timestamp'].mode())
print(df['Civil_Twilight'].mode())
print(df['Sunrise_Sunset'].mode())
print(df['Wind_Direction'].mode())
print(df['City'].mode())
print(df['Zipcode'].mode())
print(df['Airport_Code'].mode())
print(df['Timezone'].mode())
print(df['Description'].mode())


# In[ ]:


df['Weather_Condition'].fillna(str(df['Weather_Condition'].mode().values[0]), inplace=True)
df['Astronomical_Twilight'].fillna(str(df['Astronomical_Twilight'].mode().values[0]), inplace=True)
df['Nautical_Twilight'].fillna(str(df['Nautical_Twilight'].mode().values[0]), inplace=True)
df['Weather_Timestamp'].fillna(str(df['Weather_Timestamp'].mode().values[0]), inplace=True)
df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'])
df['Civil_Twilight'].fillna(str(df['Civil_Twilight'].mode().values[0]), inplace=True)
df['City'].fillna(str(df['City'].mode().values[0]), inplace=True)
df['Sunrise_Sunset'].fillna(str(df['Sunrise_Sunset'].mode().values[0]), inplace=True)
df['Wind_Direction'].fillna(str(df['Wind_Direction'].mode().values[0]), inplace=True)
df['Zipcode'].fillna(str(df['Zipcode'].mode().values[0]), inplace=True)
df['Airport_Code'].fillna(str(df['Airport_Code'].mode().values[0]), inplace=True)
df['Timezone'].fillna(str(df['Timezone'].mode().values[0]), inplace=True)
df['Description'].fillna(str(df['Description'].mode().values[0]), inplace=True)
df.isnull().sum()


# ## Aggregation and Groupby Analysis of cleaned data

# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


#Add day's column in our data.
df['day'] = df['Start_Time'].dt.day
#Add Week_day column in the data.
df['weekday'] = df['Start_Time'].dt.weekday
#Add Month column in the data.
df['month'] = df['Start_Time'].dt.month
#Add Hour column in the data.
df['hour'] = df['Start_Time'].dt.hour


# # Analyse Accident Number due to Weather Conditions.

# In[ ]:



Result1 = df.groupby('Weather_Condition').count()
Result1


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents due to Weather_Conditions')
plt.bar(Result1.index, Result1.Number, color='r')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(Result1.index, rotation='vertical', size=10)
plt.show()


# # **At what time do accidents usually occur in the US**
# ***Task Details***
# Figure out the time that accidents usually occur in the US. This can be done by using the "start time" and "end time" columns.
# 
# ***Expected Submission***
# A notebook that outputs the time that accidents usually occur.
# 
# ***Evaluation***
# A proper ranking of the times that accidents occur. The time that accidents usually occur should be at the top of the list together with the number of times accidents occurred during that specific time. Other times at lower ranks should follow. This is done to provide a more detailed solution.
# 

# ## Analyse Accident Number Per Day of the Month.

# In[ ]:


Result2 = df.groupby('day').count()
Result2


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents per day of the Month')
plt.bar(Result2.index, Result2.Number, color='b')
plt.xlabel('Days of the Month')
plt.ylabel('Number of Accidents')
plt.xticks(Result2.index, rotation='vertical', size=8)
plt.show()


# In[ ]:


print('The Day of the Month when accidents usually occur the Most in the US? : ',Result2.Number.idxmax())
Result2['Number'].sort_values(ascending=False)


# # Analyse Accident Number Per Day of the Week

# In[ ]:


Result3 = df.groupby('weekday').count()
Result3


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents per day of the Week')
plt.bar(Result3.index, Result3.Number, color='g')
plt.xlabel('Days of the Week')
plt.ylabel('Number of Accidents')
plt.xticks(Result3.index, rotation='vertical', size=8)
plt.show()


# In[ ]:


print('The Weekday when accidents usually occur the Most in the US? : ',Result3.Number.idxmax())
Result3['Number'].sort_values(ascending=False)


# # Analysis of  Accident Number per Hour of the day

# In[ ]:


Result9 = df.groupby('hour').count()
Result9


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents per Hour of the day')
plt.bar(Result9.index, Result9.Number, color='r')
plt.xlabel('Hours of the day')
plt.ylabel('Number of Accidents')
plt.xticks(Result9.index, rotation='vertical', size=20)
plt.show()


# # Analyse Accident Number Per Month.

# In[ ]:


Result8 = df.groupby('month').count()
Result8


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents Per Month')
plt.bar(Result8.index, Result8.Number, color='r')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(Result8.index, rotation='vertical', size=15)
plt.show()


# In[ ]:


print('The Month when accidents usually occur the Most in the US? : ',Result8.Number.idxmax())
Result8['Number'].sort_values(ascending=False)


# ## Ans to the Time Question

# In[ ]:


print('At what time do accidents usually occur in the US? : ',Result9.Number.idxmax())


# In[ ]:


#A proper ranking of the times that accidents occur. The time that accidents usually occur should be at the top of the list together... 
#with the number of times accidents occurred during that specific time. Other times at lower ranks should follow. 
#This is done to provide a more detailed solution.

Result9['Number'].sort_values(ascending=False)


# # Visualize US Accidents Dataset
#     This is a countrywide traffic accident dataset, which covers 49 states of the United States. The purpose of this task is to visualize the dataset states wise.

# ### Analysis of Accident Number Per State

# In[ ]:


Result4 = df.groupby('State').count()
Result4


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents per State')
plt.bar(Result4.index, Result4.Number, color='g')
plt.xlabel('States')
plt.ylabel('Number of Accidents')
plt.xticks(Result4.index, rotation='vertical', size=10)
plt.show()


# ### All states should be ranked with the first one being the state with the highest accidents. This is done for better comparisons and to provide a more detailed solution.

# In[ ]:


print('The State where accidents usually occur the Most in the US? : ',Result4.Number.idxmax())
print('The Rank of Accidents state wise is:\n ',Result4['Number'].sort_values(ascending=False))


# ### Description of the accidents that usually occur in the state of most Accidents.
# 

# In[ ]:


df1= df[df['State']=='CA']
df1.Description.unique()


# ### Accidents occurance in all the 49 States

# In[ ]:


#plotting the Lat against Long could show the map of the area
plt.figure(figsize=(50,30))
plt.title('Most Hits per Area')
plt.xlabel('Start Longitude')
plt.ylabel('Start Latitude')
plt.plot(df.Start_Lng, df.Start_Lat, ".", alpha=0.5, ms=1)
plt.show()


# In[ ]:


#plotting the Lat against Long could show the map of the area
plt.figure(figsize=(50,30))
plt.title('Most Hits per Area')
plt.xlabel('End Longitude')
plt.ylabel('End Latitude')
plt.plot(df.End_Lng, df.End_Lat, ".", alpha=0.5, ms=1)
plt.show()


# # Factors Affecting Accident Severity
#     Examine the relationship between accident severity and other accident information such as time, weather, and location.
# 

# In[ ]:


Result7 = df.groupby('Severity').count()
Result7


# In[ ]:


plt.figure(figsize=(5, 10))
plt.title('Number of Accidents Per Severity')
plt.bar(Result7.index, Result7.Number, color='r')
plt.xlabel('Severity')
plt.ylabel('Number of Accidents')
plt.xticks(Result7.index, rotation='vertical', size=15)
plt.show()


# ### Severity Relation Ships

# In[ ]:


#plt.bar(df.Severity, df.hour)
#plt.show()


# # Analysis of Accident Number Per County

# In[ ]:


Result5 = df.groupby('County').count()
Result5


# In[ ]:


plt.figure(figsize=(100, 10))
plt.title('Number of Accidents per County')
plt.bar(Result5.index, Result5.Number, color='b')
plt.xlabel('Counties')
plt.ylabel('Number of Accidents')
plt.xticks(Result5.index, rotation='vertical', size=10)
plt.show()


# In[ ]:


print('The County where accidents usually occur the Most in the US? : ',Result5.Number.idxmax())
print('The Rank of Accidents County wise is:\n ',Result5['Number'].sort_values(ascending=False))


# # Analyse Accident Number due to Weather Direction

# In[ ]:


Result6 = df.groupby('Wind_Direction').count()
Result6


# In[ ]:


plt.figure(figsize=(30, 10))
plt.title('Number of Accidents due to Wind_Directions')
plt.bar(Result6.index, Result6.Number, color='r')
plt.xlabel('Wind_Directions')
plt.ylabel('Number of Accidents')
plt.xticks(Result6.index, rotation='vertical', size=15)
plt.show()


# In[ ]:


print('The Wind_Direction for which accidents usually occur the Most in the US? : ',Result6.Number.idxmax())
print('The Rank of Accidents Wind_Direction wise is:\n ',Result6['Number'].sort_values(ascending=False))


# # Distribution of Latitudes and Longitudes

# In[ ]:


plt.hist(df.Start_Lng, bins=31, range=(-110, -65))
plt.show()


# In[ ]:


plt.hist(df.End_Lng,bins=31)
plt.show()


# In[ ]:


plt.hist(df.Start_Lat, bins=31, range=(26, 49))
plt.show()


# In[ ]:


plt.hist(df.End_Lat, bins=31)
plt.show()


# In[ ]:




