#!/usr/bin/env python
# coding: utf-8

# # Boston Crime Data Visualization & Data Analysis

# Using the Boston Crime dataset, let's try to visualize the data and do some analysis.

# ## Loading Dataset

# In[ ]:


import pandas as pd
data = pd.read_csv('../input/crime.csv', encoding="latin-1")


# ## EDA

# In[ ]:


data.head()


# The columns include an incident number, an offense code, offense group, offense description, a district, reporting area, if the incident was a shooting, the date, year, month, day of week, hour, UCR part, street, latitude, longitude, and location (essentially lat & long in a coordinate pair).

# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## Data Visualization & Analysis

# Import neccessary libraries.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ### Finding Attributes of Crime

# In this section, we'll try to find which features have distinct patterns in which crime is higher or lower.

# #### ...by district

# In[ ]:


sns.countplot(x=data['DISTRICT'],palette='viridis')


# We can see that district B2 has the highest crime rate, and district A15 has the lowest crime rate.

# #### ...by month

# In[ ]:


sns.countplot(x=data['MONTH'],palette='viridis')


# There seems to be a swell in crimes during the summer months (rising quickly from May to June, then reaching a high at July and August). Soon after, it drops. A worthy explanation is the temperature difference - the warmer months encourage crime more than colder months.

# #### ...by year

# In[ ]:


sns.countplot(x=data['YEAR'],palette='viridis')


# Interesting - it seems that the amount of crime rose dramatically from 2015-2016, stayed high through 2016-2017, and dropped significantly in 2018.

# Let's see if we can get a more detailed look by counting month.

# In[ ]:


monthList = [[],[],[],[]]

#Fill each list in monthList with 12 zeroes, one for each month
for i in range(4):
    for g in range(12):
        monthList[i].append(0)
        
#Start counting
for i in range(319073): #For every row
    monthListIndex = data.at[i,'YEAR'] - 2015 #Assigning it an index in monthList: 0, 1, 2, or 3
    monthList[monthListIndex][data.at[i,'MONTH']-1] += 1 #Finding the right bin for year & month, then adding one instance
    
#Merge each year of monthList into a continuous list
contList = []
for i in range(4):
    for g in range(12):
        contList.append(monthList[i][g])


# Let's check how many items are in contList (there should be 48; as there are 48 months in the 4 years the dataset covers)

# In[ ]:


len(contList)


# Making an x with numbers 1-48 to indicate months:

# In[ ]:


xList = []
for i in range(48):
    xList.append(i)


# In[ ]:


sns.lineplot(x=xList, y=contList)


# There is no data in the first couple months of 2015, which explains why its crime level appeared to be so low. The same is the case with the last few months of 2018. Otherwise, the data from 2015 and 2018 that are available suggest that they follow the same trend as the years with complete data.
# The data follows that pattern of a dip in winter months, usually around December-January (months 12, 24, 36), and peaking in-between.

# We learned that crime in Boston has remained about the same 2015-2018, with predictable swells and decreases in crime.

# #### ...day of week

# In[ ]:


sns.countplot(x=data['DAY_OF_WEEK'],palette='viridis')


# Uh oh! The days of week are out of order. Let's fix that by assigning each day a number, starting from Monday = 0.

# In[ ]:


dmap = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
data['DAY_OF_WEEK_#'] = data['DAY_OF_WEEK'].map(dmap)
data['DAY_OF_WEEK_#']


# In[ ]:


sns.countplot(x=data['DAY_OF_WEEK_#'],palette='viridis')


# We can see that crime remains pretty consistent across the week, peaking (but not by much) on Friday, and reaching a minimum on Sunday. One idea why crime is lower on Sunday is because of church.

# #### ...by hour

# In[ ]:


sns.countplot(x=data['HOUR'],palette='viridis')


# Below, a less pretty but continuous distplot:

# In[ ]:


sns.distplot(data['HOUR'],bins=24,hist=True,rug=True)


# In the early hours of the morning, crime quickly drops and reaches a minima at 4 or 5 AM, then clickly rising, reaching a local minima at noon and peaking at 5 PM.
# What is interesting is the sudden jump from the 23rd hour to the 24th/0th hour in crimes.

# What is also interesting is that there appears to be more crimes committed during daylight than at night.

# Using average Boston daylight hours of 6AM to 7PM, let's see how many crimes are commited for each section Day and Night.

# In[ ]:


def convert(hour):
    if hour>6 and hour<19:
        return 'Day'
    else:
        return 'Night'
data['DAY_OR_NIGHT'] = data['HOUR'].apply(convert)


# In[ ]:


sns.countplot(x=data['DAY_OR_NIGHT'],palette='viridis')


# It's confirmed that more crimes are done during the day than at night.

# #### ...by location

# Let's try to plot out location and see if there are any clusters.

# Let's change the strings a little bit to get x and y (latitude and longitude).

# In[ ]:


x = []
y = []
for i in range(319073):
    loc1 = data.at[i,'Location'].split(',')[0]
    loc1 = loc1[1:]
    loc1 = float(loc1)
    loc2 = data.at[i,'Location'].split(',')[1]
    loc2 = loc2[:-1]
    loc2 = float(loc2)
    x.append(loc1)
    y.append(loc2)    


# In[ ]:


x


# In[ ]:


y


# In[ ]:


sns.jointplot(x,y)


# The majority of points are centered around one point (as they should be - they are in the city of Boston). However, there are some outliers which definitely aren't right. Let's do some data cleaning and replace these.

# After observing the data, the Latitude should be between 42 and 43, and the Longitude should be between -70 and -72.

# In[ ]:


def process_lat(latitude):
    if latitude > 42 and latitude < 43:
        return latitude
    else:
        return np.nan
def process_long(longitude):
    if longitude < -70 and longitude > -72:
        return longitude
    else:
        return np.nan


# In[ ]:


data.head()


# In[ ]:


data['Lat'] = data['Lat'].apply(process_lat)


# In[ ]:


data['Long'] = data['Long'].apply(process_long)


# Now that we've replaced all the unreasonable outliers with NaN values, we can replace them with the average of all the values.

# In[ ]:


data['Lat'].fillna(data['Lat'].mean())


# In[ ]:


data['Long'].fillna(data['Long'].mean())


# In[ ]:


x = []
y = []
for i in range(319073):
    loc1 = data.at[i,'Lat']
    loc1 = float(loc1)
    loc2 = data.at[i,'Long']
    loc2 = float(loc2)
    x.append(loc1)
    y.append(loc2)  


# Now, we have a more zoomed-in view, since all the outliers have been removed.

# In[ ]:


sns.jointplot(x,y)


# Let' get a less crowded view with a kde plot.

# In[ ]:


sns.jointplot(x,y,kind='hex')


# We can see that the especially crime-ridden parts of the city (the darkest sections), and the better sections (more white).

# #### Conclusion/Review of Findings

# - District B2 has the highest crime rate, whereas District A15 has the lowest crime rate
# - Crime rates peak in summer months, and reach a minimum in winter months
# - Crime rates remained about the same from 2015-2018
#      - There is no data in early 2015 and late 2018
# - Crime reaches a peak on Friday and a low on Sunday
# - Crime reaches a peak at 5PM and a low at 5AM, with a sudden peak from 11PM to 12AM
# - The majority of crimes are committed during the day.

# ## Shootings

# In[ ]:


sns.countplot(data['SHOOTING'])


# Apparently none of the events were shootings.

# ## Crime

# Let's get a visual for crime amounts.

# In[ ]:


plt.figure(figsize=(20,20))
chart = sns.countplot(data['OFFENSE_CODE_GROUP'],orient='h')
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# Woah! That's a lot of types. Let's try to reduce it to the top 10.
# First, let's create a list with the values of each crime so we have access to more options.

# Sort values.

# In[ ]:


valueCounts = pd.value_counts(data['OFFENSE_CODE_GROUP'].values, sort=False)


# In[ ]:


valueCounts = valueCounts.sort_values(ascending=False)


# Get top 15.

# In[ ]:


valueCounts = valueCounts.iloc[:15]


# Convert values and indexes to lists.

# In[ ]:


indexes = valueCounts.index.values.tolist()


# In[ ]:


len(indexes)


# In[ ]:


values = valueCounts.values.tolist()


# In[ ]:


len(values)


# Make a bar plot.

# In[ ]:


plt.figure(figsize=(15,15))
chart = sns.barplot(indexes,values,palette='viridis')
chart.set_xticklabels(chart.get_xticklabels(), rotation=75)


# In[ ]:


My first kernel, thanks for reading!


# 
