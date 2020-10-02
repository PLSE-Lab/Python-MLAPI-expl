#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from collections import Counter
DF = pd.read_csv('../input/Rat_Sightings.csv')
DF.info()


# In[ ]:


DF.head()


# In[ ]:


print(DF['Bridge Highway Name'].unique()) #We don't need this
print(DF['Bridge Highway Direction'].unique())


# In[ ]:


col =['Created Date','Closed Date','Incident Address','Incident Zip','Location Type']
df = DF[col] # I take only the necessary columns 
df.head()


# In[ ]:


print(str(len(df['Incident Address'])) + ' Total recorded zip codes') # 
print(str(len(df['Incident Zip'].unique())) + ' Unique zip code')


# In[ ]:


fig, ax = plt.subplots()
df['Incident Zip'].value_counts().head(30).plot(ax=ax, kind='bar',x = 'Incident Zip', figsize=(20,7)) 
#Counting the most frequently recorded zip code


# In[ ]:


fig, ax = plt.subplots()
df['Incident Address'].value_counts().head(10).plot(ax=ax, kind='bar',x = 'Incident Address', figsize=(20,7))
#Counting the most frequently recorded address


# In[ ]:


fig, ax = plt.subplots() 
df['Location Type'].value_counts().head(10).plot(ax=ax, kind='bar',x = 'Location Type', figsize=(20,7))
#Counting the most frequently recorded location type


# In[ ]:


TRA = df.loc[df['Incident Address'] == '2131 WALLACE AVENUE'] # tabe for Incident Address = 2131 WALLACE AVENUE


# In[ ]:


TRA.head(10)


# I believe that one record of the date "Created date" == one case of the appearance of rats. If the date in the list is repeated, then there were several calls that day.

# In[ ]:


Created_dates = []
for index in TRA['Created Date'].index:
    Created_dates.append(datetime.datetime.strptime(TRA['Created Date'][index], "%m/%d/%Y %I:%M:%S %p"))


# The frequency distribution of occurrences.

# In[ ]:


Date_counts = Counter(Created_dates)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='kde') 


# We can see that cases of the reappearance of rats are very rare

# Graph the appearance of rats throughout the city. For the entire period.

# In[ ]:


Created_dates_df = []
for index in df['Created Date'].index:
    Created_dates_df.append(datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p"))
Date_counts_df = Counter(Created_dates_df)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts_df, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,20))


# We can see that the number of occurrences of rats, the minimum at the beginning of the year, then increases to the middle and then decreases again.

# The frequency distribution of occurrences of rats throughout the city. For the entire period.

# In[ ]:


Created_dates_frame.plot(kind='kde', figsize = (20,20))


# We can see that cases of the reappearance of rats are rare

# In[ ]:


df.head() #For comfort


# Graph the appearance of rats throughout the city. For the 2010 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2010 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2011 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2010 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2011 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2011 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2012 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2011 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2012 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2012 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2013 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2012 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2013 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2013 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2014 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2013 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2014 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2014 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2015 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2014 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2015 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2015 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2016 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2015 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2016 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2016 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2017 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2016 were = ' + str(quantity) )


# Graph the appearance of rats throughout the city. For the 2017 year.

# In[ ]:


Created_time = []
dateA = datetime.datetime.strptime('01/01/2017 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
dateB = datetime.datetime.strptime('01/01/2018 12:00:00 AM', "%m/%d/%Y %I:%M:%S %p")
for index in df['Created Date'].index:
    date = datetime.datetime.strptime(df['Created Date'][index], "%m/%d/%Y %I:%M:%S %p")
    if (date >= dateA and date <= dateB):
        Created_time.append(date)
Date_counts = Counter(Created_time)
Created_dates_frame = pd.DataFrame.from_dict(Date_counts, orient='index')
Created_dates_frame.plot(kind='line', figsize = (20,10))
quantity = len(Created_time) # KOLICHESTVO SLUCHEV V 2010 2011
print('Total cases in 2017 were = ' + str(quantity) )


# We see that the number of appearances of rats is increasing. We also see cases becoming more frequent during the warmer months.

# In[ ]:





# In[ ]:





# In[ ]:




