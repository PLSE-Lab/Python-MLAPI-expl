#!/usr/bin/env python
# coding: utf-8

# # analysis of the 911 calls data

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset = pd.read_csv('../input/911.csv')


# In[ ]:


dataset.info()


# In[ ]:


dataset.head(5)


# In[ ]:


# Top 5 emergencies in dataset
dataset['title'].value_counts().head(5)


# In[ ]:


# make category and sub-category column for each emergency
dataset['Category'] = dataset['title'].apply(lambda x: x.split(':')[0])
dataset['Sub-Category'] = dataset['title'].apply(lambda x: ''.join(x.split(':')[1]))


# In[ ]:


# number of calls for each category
sns.countplot('Category', data = dataset)


# In[ ]:


dataset['Category'].value_counts()


# In[ ]:


# number of unique emergency titles
dataset['title'].nunique()


# In[ ]:


# Top 5 EMS Sub categories
dataset[dataset['Category'] == 'EMS']['Sub-Category'].value_counts().head(6)


# In[ ]:


# Top 5 Fire Sub categories
dataset[dataset['Category'] == 'Fire']['Sub-Category'].value_counts().head(6)


# In[ ]:


# Top 5 Traffic Sub categories
dataset[dataset['Category'] == 'Traffic']['Sub-Category'].value_counts().head(6)


# In[ ]:


# datetime column
from dateutil import parser
dataset['datetime'] = dataset['timeStamp'].apply(lambda x : parser.parse(x))


# In[ ]:


dataset['Month'] = dataset['datetime'].apply(lambda x : x.month)
dataset['Year'] = dataset['datetime'].apply(lambda x : x.year)
dataset['day'] = dataset['datetime'].apply(lambda x : x.day)
def timeZone(timestamp):
    hour = timestamp.hour
    if (hour > 6 and hour < 12) or hour == 6:
        return 'Morning'
    elif hour == 12:
        return 'Noon'
    elif hour > 12 and hour < 17:
        return 'Afternoon'
    elif (hour > 17 and hour < 21) or hour == 17:
        return 'Evening'
    elif (hour > 21 and hour < 6) or hour == 21:
        return 'Night'
    
dataset['timezone'] = dataset['datetime'].apply(lambda x : timeZone(x))    
        


# In[ ]:


# emergency calls by time of the day, as can be seen most calls are during 
# morning and least during night
sns.countplot('timezone', data = dataset)


# In[ ]:


# column for day of the week 
import datetime
def dayofweek(x):
    day = x.weekday()
    if day == 0:
        return 'Monday'
    elif day == 1:
        return 'Tuesday'
    elif day == 2:
        return 'Wednesday'
    elif day == 3:
        return 'Thursday'
    elif day == 4:
        return 'Friday'
    elif day == 5:
        return 'Saturday'
    elif day == 6:
        return 'Sunday'
dataset['dayofweek'] = dataset['datetime'].apply(lambda x : dayofweek(x))


# In[ ]:


# emergency calls by day of week, as can be seen, Sunday has low frequency of calls
sns.countplot('dayofweek', data = dataset)


# In[ ]:


dataset['dayofweek'].value_counts()


# In[ ]:


dataset['twp'].nunique()


# In[ ]:


# from these places top 5 places from where 911 emergency calls originate are
dataset['twp'].value_counts().head(5)


# In[ ]:


# Lower Merion seems to be with most emergencies, lets take a look
plt.title('LOWER MERION incidents by Category')
sns.countplot('Category', data = dataset[dataset['twp'] == 'LOWER MERION'])


# In[ ]:


# LOWER MERION seems to be most affected by traffic emergencies, let check sub categories
dataset[(dataset['twp'] == 'LOWER MERION') & (dataset['Category'] == 'Traffic')]['Sub-Category'].value_counts()


# In[ ]:


# lets clean the data a bit 
dataset['Sub-Category'] = dataset['Sub-Category'].apply(lambda x : ''.join(x.split('-')[0]))


# In[ ]:


# it might be insightful to see more about most common problem - vehical accidents
plt.title('LOWER MERION Vehicle Accidents by timzone')
sns.countplot('timezone', data = dataset[(dataset['twp'] == 'LOWER MERION') & (dataset['Sub-Category'] == ' VEHICLE ACCIDENT')])


# In[ ]:


# this shows vehicle accidents have been most in the morning and least in the night
# to go even deeper we can see, which months had most accidents
plt.title('LOWER MERION Vehicle Accidents by month')
sns.countplot('Month', data = dataset[(dataset['twp'] == 'LOWER MERION') & 
                                      (dataset['Sub-Category'] == ' VEHICLE ACCIDENT')], palette = 'coolwarm')


# In[ ]:


# this shows accidents are maximum at time of december to february, makes sense as its the festive season
# lets check out overall accidents by month to get some ground on this
plt.title('Overall Vehicle Accidents by month')
sns.countplot('Month', data = dataset[dataset['Sub-Category'] == ' VEHICLE ACCIDENT'], palette = 'coolwarm')


# In[ ]:


# now lets have a look at the 5 places making lowest number of 911 calls
dataset['twp'].value_counts(ascending = True).head(5)


# In[ ]:


# LEHIGH COUNTY seems to be the least, lets see the emergencies
sns.countplot('Category', data = dataset[dataset['twp'] == 'LEHIGH COUNTY'], palette = 'magma')


# In[ ]:


# LEHIGH COUNTY seems to have majority of emergencies as EMS, which ones?
dataset[(dataset['twp'] == 'LEHIGH COUNTY') & (dataset['Category'] == 'EMS')]['Sub-Category'].value_counts()


# In[ ]:


# strange as this dataset has an emergency VEHICLE ACCIDENT in 2 categories EMS and Traffic


# In[ ]:


# have a look at the majoriety of places where the different categories of emergencies have occured
dataset[dataset['Category'] == 'EMS']['twp'].value_counts().head(10) #EMS category


# In[ ]:


# top 10 places for EMS category
plt.figure(figsize = (12,6))
plt.title('Top places for EMS category')
sns.countplot('twp', data = dataset[(dataset['Category'] == 'EMS') & (dataset['twp'].isin(['NORRISTOWN', 'LOWER MERION', 'ABINGTON',
                                                              'POTTSTOWN', 'LOWER PROVIDENCE', 'UPPER MERION', 
                                                              'CHELTENHAM', 'UPPER MORELAND', 'HORSHAM', 
                                                              'PLYMOUTH']))], palette = 'rainbow')
plt.xticks(rotation = 60)


# In[ ]:


dataset[dataset['Category'] == 'Fire']['twp'].value_counts().head(10) #Fire category


# In[ ]:


# top 10 places for Fire category
plt.figure(figsize = (12,6))
plt.title('Top places for Fire category')
sns.countplot('twp', data = dataset[(dataset['Category'] == 'Fire') 
                                    & (dataset['twp'].isin(['LOWER MERION', 
                                        'ABINGTON', 'NORRISTOWN', 
                                        'CHELTENHAM', 'POTTSTOWN', 'UPPER MERION', 
                                        'WHITEMARSH', 'UPPER PROVIDENCE', 
                                        'LIMERICK', 'PLYMOUTH']))], palette = 'rainbow')
plt.xticks(rotation = 60)


# In[ ]:


dataset[dataset['Category'] == 'Traffic']['twp'].value_counts().head(10) #Traffic category


# In[ ]:


# top 10 places for Traffic category
plt.figure(figsize = (12,6))
plt.title('Top places for Traffic category')
sns.countplot('twp', data = dataset[(dataset['Category'] == 'Traffic') 
                                    & (dataset['twp'].isin(['LOWER MERION', 'UPPER MERION', 'ABINGTON', 
                                                            'CHELTENHAM', 'PLYMOUTH', 'UPPER DUBLIN', 
                                                            'UPPER MORELAND', 
                                                            'HORSHAM', 'MONTGOMERY', 'NORRISTOWN']))], 
              palette = 'rainbow')
plt.xticks(rotation = 60)


# In[ ]:


# there are many things that can be infered from above graphs. Norristown has high number of EMS, 
# while low number of Fire and Traffic
# On the oter hand Lower Merion has almost similar count for all categories


# In[ ]:


# LOWER MERION has most fire calls
dataset[(dataset['Category'] == 'Fire') & (dataset['twp'] == 'LOWER MERION')]['addr'].value_counts()


# In[ ]:


# SCHUYLKILL EXPY & CONSHOHOCKEN STATE UNDERPASS seems to be most affected region by fire
sns.countplot('Sub-Category', data = dataset[((dataset['addr'] == 'SCHUYLKILL EXPY & CONSHOHOCKEN STATE UNDERPASS') | 
          (dataset['addr'] == 'SCHUYLKILL EXPY & WAVERLY RD UNDERPASS') |
           (dataset['addr'] == 'SCHUYLKILL EXPY & MILL CREEK RD UNDERPASS')) & 
            (dataset['Category'] == 'Fire')], palette = 'magma')
plt.xticks(rotation = 60)


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot('Sub-Category', data = dataset[dataset['Category'] == 'Fire'], palette = 'magma')
plt.xticks(rotation = 90)


# In[ ]:


# fire alarms cause most fire calls
# place from where most fire alarms come
plt.figure(figsize = (12,6))
sns.countplot('twp', data = dataset[dataset['Sub-Category'] == ' FIRE ALARM'])
plt.xticks(rotation = 80)


# In[ ]:


# most fire alarms also are in LOWER MERION

