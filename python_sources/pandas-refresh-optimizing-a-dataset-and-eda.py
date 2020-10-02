#!/usr/bin/env python
# coding: utf-8

# [The Stanford Open Policing Project](https://openpolicing.stanford.edu/)
# 
# On a typical day in the United States, police officers make more than 50,000 traffic stops. The project collects and standardize data on vehicle and pedestrian stops from law enforcement departments across the country. The current dataset is a subset comprising only the Rhode Island County. 
# 
# The current analysis is based on the Course - [Analyzing Police Activity with Pandas](  https://campus.datacamp.com/courses/analyzing-police-activity-with-pandas)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling
import matplotlib.pyplot as plt
from pprint import pprint as pp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Preparing the data for analysis
# 
# 
# Before beginning your analysis, it is critical that you first examine and clean the dataset, to make working with it a more efficient process. Let's practice fixing data types, handling missing values, and dropping columns and rows while learning about the Stanford Open Policing Project dataset.

# In[ ]:


INPUT = '/kaggle/input/stanford-open-policing-project/police_project.csv'


# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 25)

df = pd.read_csv(INPUT)

df.head()


# In[ ]:


df.info()


# The [pandas_profiling](https://pandas-profiling.github.io/pandas-profiling/docs/) eliminates a lot of the pain in EDA. 

# In[ ]:


df.profile_report(style={'full_width':True})


# The `pandas_profiling` report did the most heavy lifting for us, but let's try to replicate from scratch some of it as an exercise.

# In[ ]:


df.shape


# The method `df.isnull()` creates a dataframe of boolean values, where `True` is for missing values and `False`otherwise. Since in python `True == 1` and `False == 0` we can count the missing values by summing this dataframe of boolean values. This will return a series counting the missing values per column.

# In[ ]:


df.isnull().sum()


# ## Dropping Columns
# 
# 1. The column `county_name` is exclusively missing values, maybe because this is a subset from the original dataset comprising one specific state `Rhode Island`, so this column has no information and we can safely drop it.

# In[ ]:


df.drop('county_name', axis='columns', inplace=True)


# ## Dropping Rows
# One interesting analysis is to compare outcomes by gender, so the gender information might be critical and it makes sense to drop the rows missing the gender, since the number is small `5.8%`

# In[ ]:


df.dropna(subset=['driver_gender'], inplace=True)
df.isnull().sum()


# After eliminating the records with `NULL` values in `driver_gender`,  we pretty much eliminated almost all `NULL` values in the dataset, let's fill the remaining `NULL` values in `driver_age` with the average age so we can properly define the datatypes.

# In[ ]:


df.loc[df.driver_age.isnull(), ['driver_age']] = int(df.driver_age.mean())
df.loc[df.driver_age_raw.isnull(), ['driver_age_raw']] = int(df.driver_age_raw.mean())


# ## Proper Datatypes
# The `pandas_profiling` also gaves us everything we need to define the correct datatypes.
# Pandas infer datatypes upon loading but usually is not very efficient, datatypes are important because it affects which operations can be performed on it and usually we can be way more efficient (i.e. less memory and faster loading types) if we properly set the datatypes. The current dataframe occupies `8.7MB` in memory, let's see how we can lower it only by defining the correct datatypes.

# In[ ]:


df.info()


# * `object`: Python objects  - Strings, Lists, etc
#     * Large Space and limited operations, avoid whenever possible
# * `bool`: `True` and `False` values. 
#     * 1 byte - Logical and Mathematical Operations
# * `int, float`
#     * enables math operations
# * `datetime`
#     * enables different date attributes such as day, month, year 
#     * methods to slice and resample, not possible with strings
# * `category`
#     * uses less memory and runs faster
# 

# In[ ]:


# When assigning to columns, only the square brackets notation works.
df['is_arrested'] = df.is_arrested.astype('bool')
df.info()


# * The `driver_age`goes from 15 to 88, so a `int8/uint8` ( 1 byte of memory, range between -128/127 or 0/255 ) is sufficient.
# * The `driver_age_raw` are year values so a `int16/float16` (2 bytes of memory, range between -32768 and 32767 or 0/65535) is sufficient. 
# 
# [More info here](https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e)

# In[ ]:


df['driver_age'] = df['driver_age'].astype('int8')
df['driver_age_raw'] = df['driver_age_raw'].astype('int16')
df.info()


# For the remaining categorical datatypes:

# In[ ]:


for col in ['driver_gender', 'driver_race', 'violation_raw', 'violation', 'stop_outcome']:
    df[col] = df[col].astype('category')
    
df.info()


# By just defining the proper datatypes we went from `8.7MB` to `4.2MB` 

# ## Creating a Datetime Index
# Let's combine the `stop_date` and `stop_time` columns to create a `datetime` column. In the datetime format the date and time are separated by a space.

# In[ ]:


datetime = df.stop_date.str.cat(df.stop_time, sep= ' ')
df['stop_datetime'] = pd.to_datetime(datetime)


# Now that we have a datetime column, we can define it as the DataFrame's index. By doing this we can easily slice, subset and analyze the data by date and time.

# In[ ]:


df.set_index('stop_datetime', inplace=True)
df.head()


# In[ ]:


# Now we can drop the redundant columns
df.drop(['stop_date', 'stop_time'], axis='columns', inplace=True)
df.head()


# # Analysing Outcomes by Gender

# ## Counting Unique values
#  * `value_counts()`: returns a Series with the distint unique values for each column

# In[ ]:


df.stop_outcome.value_counts()


# In[ ]:


# Percentage from the total
df.stop_outcome.value_counts() / df.shape[0]


# In[ ]:


# As usual there is a method which do that for us
df.stop_outcome.value_counts(normalize=True)


# In[ ]:


df.driver_race.value_counts()


# ## Filtering by Gender

# In[ ]:


male = df[df.driver_gender == 'M']
female = df[df.driver_gender == 'F']

print("Female Violations")
pp(female.violation.value_counts(normalize=True))

print("\nMale Violations")
pp(male.violation.value_counts(normalize=True))


# About two-thirds of female traffic stops are for speeding, whereas for males is about half. This doesn't mean that females speed more often than males, however, since we didn't take into account the number of stops or drivers.

# In[ ]:


print(f"Female Records: {female.shape[0]}\nMale Records: {male.shape[0]}")


# ## Filtering by multiple Conditions
# When filtering by multiple conditions, inside the brackets the conditions are enclosed by parenthesis and the logical operators are *C* like instead:
# * `&`: represents the `and` operation
# * `|`: represents the `or` operation

# In[ ]:


arrested_females = df[(df.driver_gender == 'F') & (df.is_arrested == True)]
arrested_males = df[(df.driver_gender == 'M') & (df.is_arrested == True)]


# In[ ]:


print(f"Arrested Females: {arrested_females.shape[0]}\nArrested Males: {arrested_males.shape[0]}")


# ## Speeding outcomes
# 
# When a driver is pulled over for speeding, many people believe that gender has an impact on whether the driver will receive a ticket or a warning. 

# In[ ]:


female_and_speeding = df[(df.driver_gender == 'F') & (df.violation == 'Speeding')]
male_and_speeding = df[(df.driver_gender == 'M') & (df.violation == 'Speeding')]

print("Female Outcomes After Speeding")
print(female_and_speeding.stop_outcome.value_counts(normalize=True))
print("\nMale Outcomes After Speeding")
print(male_and_speeding.stop_outcome.value_counts(normalize=True))


# The Proportions of Citations ( Ticket ) and Warning doesn't seem any different for different genders.

# ## Search Rate
# 
# During a traffic stop, the police officer sometimes conducts a search of the vehicle. In this exercise, you'll calculate the percentage of all stops that result in a vehicle search, also known as the search rate.

# In[ ]:


print(df.search_conducted.dtype)
print(df.search_conducted.value_counts(normalize=True))

print("\nPercentage of Searched Vehicles:")
print(f'{df.search_conducted.mean() * 100:.2f}%')


# In[ ]:


print("\nPercentage of Searched Vehicles (Female):")

print(f"{df[df.driver_gender == 'F'].search_conducted.mean() * 100:.2f}%")

print("\nPercentage of Searched Vehicles (Male):")

print(f"{df[df.driver_gender == 'M'].search_conducted.mean() * 100:.2f}%")


# A better way to do this is aggregate using the groupby method:

# In[ ]:


print("\nSearched Vehicles by gender:")

df.groupby('driver_gender').search_conducted.mean()


# ## Adding a second factor to the analysis
# 
# Even though the search rate for males is much higher than for females, it's possible that the difference is mostly due to a second factor.
# 
# For example, you might hypothesize that the search rate varies by violation type, and the *difference in search rate between males and females is because they tend to commit different violations.*
# 
# You can test this hypothesis by examining the search rate for each combination of gender and violation. If the hypothesis was true, you would find that males and females are searched at about the same rate for each violation

# In[ ]:


print(df.groupby(['violation', 'driver_gender']).search_conducted.mean())


# The search rate is higher for males than for females for all types of specified violations, disproving our hypothesis, at least for this dataset.

# ## Protective Frisks
# During a vehicle search, the police officer may pat down the driver to check if they have a weapon. This is known as a "protective frisk."

# In[ ]:


print(df.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
df['frisk'] = df.search_type.str.contains('Protective Frisk', na=False)

# Take the sum of 'frisk'
print(df.frisk.sum())


# In[ ]:


searched = df[df.search_conducted == True]

print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())


# Males are frisked more often than females, though we can't conclude that this difference is caused by the driver's gender.

# # Adding some Visualizations
# ## Calculating the hourly arrest rate
# 
# When a police officer stops a driver, a small percentage of those stops ends in an arrest. This is known as the arrest rate. Let's check whether the arrest rate varies by time of day.
# 
#     0 = midnight
#     12 = noon
#     23 = 11 PM
# 

# In[ ]:


print(f"Mean of Arrests: {df.is_arrested.mean():.3f}")
print("Hourly Arrest Rates")
hourly_arrest_rate = df.groupby(df.index.hour).is_arrested.mean()
pp(hourly_arrest_rate)


# In[ ]:


hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()


# The arrest rate has a significant spike overnight, and then dips in the early morning hours.

# ## Plotting drug-related stops
# 
# In a small portion of traffic stops, drugs are found in the vehicle during a search.
# 
# The Boolean column drugs_related_stop indicates whether drugs were found during a given stop. 
# 
# Let's resample this column by year to see the trend

# In[ ]:


annual_drug_rate = df.drugs_related_stop.resample("A").mean()

annual_drug_rate.plot()
plt.xlabel('Year')
plt.ylabel('Drug Found Rate')
plt.title('Yearly Drug Related Stops')
plt.show()


# The rate of drug-related stops nearly double in 10 years - 2005-2015
# 
# Let's see if this increase is correlated with the search rate.

# In[ ]:


annual_search_rate = df.search_conducted.resample('A').mean()

annual = pd.concat([annual_drug_rate, annual_search_rate], axis=1)

annual.plot(subplots=True)
plt.xlabel('Year')
plt.ylabel('Annual Rate')
plt.title('Yearly Searchs and Drug Related Stops')
plt.show()


# Actually the `search_conducted` appears to be inversally correlated, contrary to our hypotheses

# ## Violations by Race
# *  * **Frequency Table** It shows how many times each combination of values occurs

# In[ ]:


table = pd.crosstab(df.driver_race, df.violation)
table


# In[ ]:


table = pd.crosstab(df.driver_race, df.violation, normalize=True)
table


# In[ ]:


table.plot(kind='barh')
plt.show()


# All the numbers for `Whites` are higher because there are a lot more stops of whites, probably because there are a lot more of them. We can't conclude much from this race difference, this plot was only to ilustrate the use of Frequency Table.

# ## How long might you be stopped for a violation?
# The `stop_duration` column tells you approximatelly how long the driver was detained by the officer. Since the durations are stored as strings and it's an approximation, we must map to an estimated number to get any useful insight.
# 
# <ul>
# <li>Convert <code>'0-15 Min'</code> to <code>8</code></li> 
# <li>Convert <code>'16-30 Min'</code> to <code>23</code></li>
# <li>Convert <code>'30+ Min'</code> to <code>45</code></li>
# </ul>

# In[ ]:


print(df.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min': 8, '16-30 Min': 23, '30+ Min': 45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
df['stop_minutes'] =df.stop_duration.map(mapping)
print(df.stop_minutes.unique())


# There are 2 single records, '2' and '1', which are not clear what they mean, so we don't map them. This turns them into `nan` so we can safely discard.  

# In[ ]:


df.dropna(subset=['stop_minutes'], inplace=True)


# In[ ]:


stop_length = df.groupby('violation_raw').stop_minutes.mean()
stop_length.sort_values().plot(kind='barh')

plt.xlabel('Approximate Duration in Minutes')
plt.ylabel('Detailed Violation')
plt.title("Stopping Duration by Violation")
plt.show()


# That's it for now. It's just a little exploration of data using just pandas. 
