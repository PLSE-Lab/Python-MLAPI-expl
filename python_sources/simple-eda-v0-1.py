#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction

# ## 1.1 Loading packages and Attributes of data
# 
# All packages used for analysis will be loaded, including representative packages such as `numpy` and` pandas`. I put a simple comment next to the package.

# In[ ]:


import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization
import os # Load file


# All necessary packages have been loaded. Let's check the data properties first.
# 
# * `parse_dates` : makes the type of column datetime

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv', parse_dates=['datetime'])
df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', parse_dates=['datetime'])
df_submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv', parse_dates=['datetime'])


# In[ ]:


df_train.head(3)


# In[ ]:


print("1) train data shape : ", df_train.shape)
print("2) train data type : ")
print(df_train.dtypes)


# The variable names and class types are the same as above, with a total of 1,10,886 observations and 12 variables. A brief description of the variable follows.
# 
# Variable name  | Description
# ---------------|----------------
# datetime       | hourly date + timestamp 
# season         | main periods into which a year (1: spring, 2: summer, 3: fall, 4: winter)
# holiday        | whether the day is considered a holiday
# workingday     | whether the day is neither a weekend nor holiday
# weather        | condition of the atmosphere
#                | (1: Clear, Few clouds, Partly cloudy, Partly cloudy
#                |  2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#                |  3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#                |  4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog)
# temp           | temperature in Celsius
# atemp          | "feels like" temperature in Celsius
# humidity       | relative humidity
# windspeed      | wind speed
# casual         | number of non-registered user rentals initiated
# registered     | number of registered user rentals initiated
# count          | number of total rentals

# In[ ]:


df_test.head(3)


# In[ ]:


print("1) test data shape : ", df_test.shape)
print("2) test data type : ")
print(df_test.dtypes)


# In[ ]:


df_submission.head(3)


# In[ ]:


print("1) submission data shape : ", df_submission.shape)
print("2) submission data type : ")
print(df_submission.dtypes)


# ## 1.2 Check Dataset

# ### 1.2.1 Redundancy check
# * `duplicated` : Check for duplication

# In[ ]:


sum(df_train.duplicated()), sum(df_test.duplicated())


# ### 1.2.2 Missing values
# 
# Training data and test data do not have null values.

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# ### 1.2.3 Divide datetime into details
# 
# Extract the year, month, day, hour, minute, and second from datetime and make each new column into a column.

# In[ ]:


df_train['year'] = df_train['datetime'].dt.year # year
df_train['month'] = df_train['datetime'].dt.month # month
df_train['day'] = df_train['datetime'].dt.day # day
df_train['hour'] = df_train['datetime'].dt.hour # hour
df_train['minute'] = df_train['datetime'].dt.minute # minute
df_train['second'] = df_train['datetime'].dt.second # second


# In[ ]:


df_train.head(3)


# ----------------

# # 2. Exploratory Data Analysis

# ## 2.1 Target distribution

# In[ ]:


plt.hist(df_train['count'], bins=15)
plt.xlabel('Count', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)


# In[ ]:


sns.boxplot(data=df_train, y='count')


# ## 2.2 Average number of total rentals (Season, Hour, Working day)
# 
# * **Season** : Highest rentals in Q3.
# * **Hour** : Average number of total rentals is hightes at 8 o'clock and 17~18 o'clock during rush hour.
# * **Working day** : Weekdays are higher than weekends and holidays. (1 > 0)

# In[ ]:


figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18, 10)

sns.boxplot(data=df_train, x='season', y='count', ax=ax1)
sns.boxplot(data=df_train, x='hour', y='count', ax=ax2)
sns.boxplot(data=df_train, x='workingday', y='count', ax=ax3)


# ## 2.3 Average number of total rentals (Year, Month, Day, Hour, Minute, Second)
# 
# * **Year** : Average number of total rentals increased in 2012 compared to 2011 (2012 > 2011)
# * **Month** : June is the largest monthly rental period and average number of total rentals are mainly high during the warm season (May~Oct)
# * **Day** : There is no noticeable difference in daily rentals. Also, there are only 1 ~ 19 days in train data, and there are 20 ~ 31 days of test data. So, I think day variable cannot be used as a feature.
# * **Hour** : Average number of total rentals is hightes at 8 o'clock and 17~18 o'clock during rush hour.
# * **Minute, Second** : All minutes and seconds are 0. So cannot be used.

# In[ ]:


figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 12)

sns.barplot(data=df_train, x="year", y="count", ax=ax1)
sns.barplot(data=df_train, x="month", y="count", ax=ax2)
sns.barplot(data=df_train, x="day", y="count", ax=ax3)
sns.barplot(data=df_train, x="hour", y="count", ax=ax4)
sns.barplot(data=df_train, x="minute", y="count", ax=ax5)
sns.barplot(data=df_train, x="second", y="count", ax=ax6)

ax1.set(title="Rental amounts by year")
ax2.set(title="Rental amounts by month")
ax3.set(title="Rental amounts by day")
ax4.set(title="Rental amounts by hour")
ax5.set(title="Rental amounts by minute")
ax6.set(title="Rental amounts by second")


# ## 2.4 Bicycle rental by time zone (by Working day, Day of week, Season, Weather)
# 
# Extract Day of the week variable from datetime and make new column.
# * Monday: 0 / Tuesday: 1 / Wednesday: 2 / Thursday: 3 / Friday: 4 / Saturday: 5 / Sunday: 6

# In[ ]:


df_train['dayofweek'] = df_train['datetime'].dt.dayofweek
df_train.head(3)


# In[ ]:


df_train['dayofweek'].value_counts()


# * **Working day** : There are a lot of rentals during commuting hours on working days and a lot of rentals during daytime on non-working days.
# * **Day of week** : There are a lot of rentals during commute hours from Monday to Friday, and a lot of rentals during daytime on Saturdays and Sundays.
# * **Season** : The rental amount is the lowest in January to March, and the rental amount is the highest in July to September.
# * **Weather** : The better the weather, the more the rental, and when there is heavy rain, there is almost no rental.

# In[ ]:


fig,(ax1, ax2, ax3, ax4)= plt.subplots(nrows=4)
fig.set_size_inches(18,25)

sns.pointplot(data=df_train, x='hour', y='count', hue='workingday', ax=ax1)
sns.pointplot(data=df_train, x='hour', y='count', hue='dayofweek', ax=ax2)
sns.pointplot(data=df_train, x='hour', y='count', hue='season', ax=ax3)
sns.pointplot(data=df_train, x='hour', y='count', hue='weather', ax=ax4)

