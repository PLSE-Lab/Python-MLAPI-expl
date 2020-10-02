#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Samsung Health Step Data
# 
# This notebook is a quick analysis of my daily steps over the last two years as recorded by the Samsung Health app on my Android phone
# 
# If you want to do the same, this file is obtained from the Samsung Health App  
# Go to Options (top right) -> Settings -> Download Personal Data -> Download  
# You should see a file in the folder downloaded that looks like: 'com.samsung.shealth.step_daily_trend.YYYYMMDDHHMM.csv'  
# There's lots other stuff in there, depending which features of the app you use.
# 

# First let's import the things we'll be needing:

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#Gets rid of the seaborn 'FutureWarning'


# ## Importing and Cleaning the Data

# Table starts on 2nd row so use header=1

# In[ ]:


steps = pd.read_csv('../input/com.samsung.shealth.step_daily_trend.201812071026.csv', header=1 ) 
steps.head()


# The 'day_time' format looks like unix time (milliseconds since Epoch)  
# Let's convert it to something more useable

# In[ ]:


steps['day_time'] = pd.to_datetime(steps['day_time'], unit='ms')
steps.head()


# There seems to be a duplicate for each entry,with source_type = -2 or -0.  
# Let's get just the 0s.  
# There's also more than one log per day, so let's sum those  
# Then let's drop the source_type, speed and calories

# In[ ]:


steps = steps[steps['source_type']==0].groupby('day_time').sum()
steps.drop(['source_type','speed', 'calorie'], axis=1, inplace=True)
steps.reset_index(inplace=True)
steps.head()


# We can create new columns for month,date,year extracted from the time stamp

# In[ ]:


steps['Day'] = steps['day_time'].apply(lambda datestamp: datestamp.day)
steps['Year'] = steps['day_time'].apply(lambda datestamp: datestamp.year)


#Functions to get days of the week and months as strings instead of indexes 
#(0-6 and 0-12 respectively)

def dayofweek(datestamp):
    return ['Mon', 'Tue', 'Wed','Thur','Fri','Sat','Sun'][datestamp.weekday()]
def monthname(datestamp):
    return ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datestamp.month-1]

steps['Weekday'] = steps['day_time'].apply(lambda datestamp: dayofweek(datestamp)) 
steps['MonthName'] = steps['day_time'].apply(lambda datestamp: monthname(datestamp))

#Keeping the month as an index to construct the yearmonth column below:

steps['Month'] = steps['day_time'].apply(lambda datestamp: datestamp.month)

#Function to get a combined 'YearMonth' column

def yearmonth(cols):
    month=cols[0]
    year=cols[1]
    return '{}-{}'.format(month, year)

steps['YearMonth'] = steps[['Month', 'Year']].apply(lambda cols: yearmonth(cols), axis=1)

steps.head()


# Now we can get rid of the day_time column since we have extracted all the information into other columns.  
# Our final DataFrame looks like :

# In[ ]:


steps.drop('day_time', inplace=True, axis=1)

steps.head()


# ## Analysis and Visualization

# Let's investigate the data:  
#   
# Firstly, how many years' worth of data do we have?

# In[ ]:


print ('Number of years = {}'.format(steps['count'].count()/365))


# Just over 2 year's of data  
# What does the distribution of the data look like?

# In[ ]:


steps['count'].describe()


# So my all time average is 8810 steps, with a standard deviation of 5587.  
# On half of all days I walked more than 7761 steps.  
# 
# My record was 38652 steps in one day.   
# If I recall correctly, I was visiting Budapest!  
# 
# When was that again? :  

# In[ ]:


steps[steps['count']==steps['count'].max()][['Day','MonthName', 'Year']]


# A look at the distribution of the number of daily steps

# In[ ]:


sns.distplot(steps['count'], bins=25).set(xlim=(0,steps['count'].max()))  


# Was I more active in 2017 or 2018?

# In[ ]:


plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day in 2017 and 2018')
sns.barplot(x='Year', y=steps['count'], data=steps[steps['Year']>2016])


# 2017 wins!  
# Let's break that down by month. 

# In[ ]:


plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day for each month, 2017 vs 2018')
sns.barplot(x='MonthName', y='count', data=steps[steps['Year']>2016], hue='Year', order = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] )


# Looks like i'm often walking a lot while on holiday!  
# And I went on holiday in July in 2017 vs August in 2018  
# 
# Am I more active at the start or the end of a typical month? (Average over years)

# In[ ]:


plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day over a typical month')
sns.barplot(x='Day', y='count', data=steps)


# Doesn't seem to be a factor.  
# 
# What about activity through a typical week?  

# In[ ]:


plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day over a typical week')
sns.barplot(x='Weekday', y='count', data=steps, order = ['Mon', 'Tue', 'Wed','Thur','Fri','Sat','Sun'], palette = 'deep')


# Momentum building towards the weekend with a lazy Sunday!  
# 
# Let's look at the year 2017 in detail:

# In[ ]:


piv = steps[steps['Year']==2017].pivot_table(index='Month',columns='Day', values='count').fillna(0)
plt.figure(figsize=(20,5))
plt.title('Steps in 2017')
sns.heatmap(piv, cmap='viridis')


# If you look closely you can spot the holiday periods and weekends.  
# 
# What about the total distance walked and total steps per year?

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Total Distance (km)')
sns.barplot(x='Year', y=steps['distance']/1000, data=steps[steps['Year']>2016], estimator= sum)


# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Total Steps (in millions)')
sns.barplot(x='Year', y=steps['count']/1000000, data=steps[steps['Year']>2016], estimator= sum)


# Speaking of distance walked, how far exactly is 1 step?

# In[ ]:


sns.scatterplot(x='count', y='distance', data=steps)


# Let's get the correlation coefficient (i.e. the distance per step : step-size)

# In[ ]:


from sklearn.linear_model import LinearRegression  #For step size regression fit

lm= LinearRegression()
x=steps['count'].values.reshape(-1, 1)
y=steps['distance'].values.reshape(-1, 1)
lm.fit(x,y)
print('Average step size = {:.2f} cm'.format(lm.coef_[0][0]*100))


# How does this compare to a simple average, taken as total distance / total steps?  

# In[ ]:


total_distance = steps['distance'].sum()
total_steps =  steps['count'].sum()
average_step = total_distance/total_steps
print('Average step size = {:.2f} cm'.format(average_step))


# This is different to the value as calculated via regression. Not totally sure why...  
# 
# Let's investigate "lazy days" when my steps are less than some threshold

# In[ ]:


thresh=1000
lazydays= steps[steps['count']<thresh]['count'].count()
print('{} days where steps < {}'.format(lazydays,thresh))


# We could make a cumulative steps function which returns the number of days with at least 'n' steps  
# e.g. My phone has a 'target reached!' notification when I go through 10,000 steps  
# which I have received at_least(10000) times :

# In[ ]:


def at_least(n):
    return steps[steps['count']>n]['count'].count()

print (at_least(10000))


# Let's plot that function for every value of steps recorded (laziest day to busiest day)

# In[ ]:


print('Laziest day = {}'.format(steps['count'].min()))
print('Busiest day = {}'.format(steps['count'].max()))
sns.lineplot(x= steps['count'], y= steps['count'].apply(lambda values : at_least(values)))


# Thanks for taking a look!
