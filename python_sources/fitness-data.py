#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# On manual analysis of the dataset, there are some inaccurate values are there. 
# 
# more steps the person took means the more distance should be covered, but some inaccuracy are there and also the more steps means more calories burn which also not matching in some days. 
# The reason for this issue may be the fit band used may be too old. or the band is inaccurate. 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


fit = pd.read_csv('/kaggle/input/one-year-of-fitbit-chargehr-data/One_Year_of_FitBitChargeHR_Data.csv')
fit.head()


# In[ ]:


fit['Minutes_sitting'] = fit['Minutes_sitting'].astype(int)


# creating a new column, to get the total activity time of the person and also how long wear/used his/her band

# In[ ]:


fit['total_activity'] = fit[['Minutes_of_slow_activity','Minutes_of_moderate_activity','Minutes_of_intense_activity']].sum(axis=1)
fit['total_day_minutes'] = fit['Minutes_sitting'] + fit['total_activity']
fit.head()


# Dropping the column 'floor' which we will never use

# In[ ]:


fit.drop(columns = 'floors',inplace=True)


# In[ ]:


fit.info()


# In[ ]:


fit.describe()


# In[ ]:


#Converting the date from object to date format  
fit['Date'] = pd.to_datetime(fit['Date'],format='%d-%m-%Y')
#Extracting the month only
fit['Month_only'] = pd.to_datetime(fit['Date']).dt.month
fit


# In[ ]:


### Scatterplot of Calories Vs total activity

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='total_activity', y='Calories', data=fit)

ax.set_title('Scatterplot of calories and total_activities')


# From the above graph we can confirm that when the activity increases, then calories burn also will increase

# In[ ]:


#Bar plot with respect ot date and calories burned
plt.figure(figsize=(20,6))
sns.barplot(x="Date", y="Calories", data=fit)
plt.title('Calories with respect to date')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#Extracting the week day from the date
fit['day_of_week'] = fit['Date'].dt.dayofweek


# In[ ]:


plt.figure(figsize=(15,6))
data = fit.groupby('day_of_week').sum().reset_index()
sns.barplot(x='day_of_week',y='total_activity',data=fit)
plt.title('DAY OF THE WEEK')
plt.show()


# From the above bar plot we can see that the user activity is low in saturday and high in friday.

# In[ ]:


plt.figure(figsize=(15,6))
sns.scatterplot(x='Date', y='total_activity', data=fit)
plt.title('Activity')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
data = fit.groupby('Month_only').sum().reset_index()
sns.barplot(x='Month_only',y='total_activity',data=fit)
plt.title('Month vs activity')
plt.show()


# The user activity was high on april and low on october

# In[ ]:


#Distanc was in object and also it has ' , ' in it so here I am removing the comma and converting it as float
fit['Distance'] = fit.Distance.str.replace(',', '').astype(float)
fit['Distance']
#pd.to_numeric(cpy_fit, downcast='integer')
#cpy_fit['first'] = cpy_fit['first'].astype(int)

#cpy_fit["Last"] = cpy_fit["Last"].astype(str).astype(int)
#cpy_fit['Distance_meter'] = cpy_fit['first'] * 1000
#cpy_fit['Last']  = cpy_fit['Last'].astype(int)
#cpy_fit['Distance_meter'] = cpy_fit['Distance_meter'] + cpy_fit['Last']


# In[ ]:


#converting values into meters already it was in KM (already 2places of decimal was correct, multiplying by 10),in above cell when we convert into float it changed
fit['Distance'] = fit['Distance']*10
fit


# Here trying to findout the person is men or women.
# 

# In[ ]:


fit['length'] = fit['Distance']/(fit['Steps']*1000)
fit['length']


# In[ ]:


fit['length'].median()


# From the above calculation the pedometer set some standard values.
# An average man's stride length is 78 centimeters, while a woman's average stride length is 70 centimeters.
# [https://livehealthy.chron.com/determine-stride-pedometer-height-weight-4518.html](http://)
# 
# ### **from our calculation the values 75cm it seems the person is male**

# In[ ]:


print(fit['total_day_minutes'].mean())
print(fit['Minutes_sitting'].mean())
print(fit['total_activity'].mean())


# * The user had tied his band on average of 862 minutes per day. that is 14 hours a day.
# 
# * The user sits on average of 563 minutes per day, that is 9 hours a day, the user may working on white collar job or the band counts his sleeping time also as sitting time. But on average the user uses his band nearly 14 hours only which seems he never his band on sleeping time and also there is no sleeping time in the data set. so we can assume that the user uses his watch only on day time. 
# * On this assumption, the user spends most of his time in sitting or doing very light works. the time spent on sitting is very high.
# 
# * The total activity of the user on average is 298 minutes per day, that is 5 hours a day. and also I assume that the user travel from his home to office. most of the time is spent on slow activity.

# In[ ]:


descending = fit.sort_values(by='Distance', ascending=False)
descending.head(10)


# In[ ]:



# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Distance', y='total_activity', data=descending)

ax.set_title('Scatterplot of distance and total_activities')


# In[ ]:


ascending = fit.sort_values(by='Distance')
ascending.head(20)


# from the above table, we can find that there is no activity on 1st 3 days, the user may forget to use his band.
# 
# and the remaining day seems the user may be ill or he has some urgent work in his office. also some of the day are fall on weekends too

# In[ ]:


# number of days the user walk less than 1 km in a day.
ascending[ascending['Distance'] < 1000].groupby('Distance').sum()


# Lets try to finouthis walking speed.
# we know that formula for speed is distance/time. 
# the distance is in meter and the time is in minutes.

# In[ ]:


fit.head()


# In[ ]:



fit['speed_km'] = (fit['Distance']/1000)/(fit['total_activity']/60)
print(fit['speed_km'].median(), 'kilometer per hour')


# In[ ]:


cpy_fit =fit
cpy_fit = cpy_fit.drop(['Month_only', 'day_of_week', 'length','speed_km'], axis = 1)
f, ax = plt.subplots(figsize=(10, 8))
corr_temp = cpy_fit.corr()
ax = sns.heatmap(corr_temp, mask=np.zeros_like(corr_temp, dtype=np.bool), 
                 cmap=sns.diverging_palette(220, 10, as_cmap=True),
                 annot=True, square=True)

ax.set_title('Correlation between calories and different activities')


# The pesons walks on the speed of 1.6 km/hr which is too slow, but there may be chance for inaccuracy in fitband.
# [https://greatist.com/health/average-walking-speed#men-vs-women](http://)

# ## Conclusion
# * the user activity is low in saturday and high in friday.
# * The user activity was high on april and low on october.
# * From our calculation the values 76cm it seems the person is male
# * The user had tied his band on average of 862 minutes per day. that is 14 hours a day.
# 
# * The user sits on average of 563 minutes per day, that is 9 hours a day, the user may working on white collar job or the band counts his sleeping time also as sitting time. But on average the user uses his band nearly 14 hours only which seems he never his band on sleeping time and also there is no sleeping time in the data set. so we can assume that the user uses his watch only on day time.
# 
# * On this assumption, the user spends most of his time in sitting or doing very light works. the time spent on sitting is very high.
# 
# * The total activity of the user on average is 298 minutes per day, that is 5 hours a day. and also I assume that the user travel from his home to office. most of the time is spent on slow activity.
# 
# * The user didnt complete even 1km for 23 days.
# 
# * The pesons walks on the speed of 1.6 km/hr which is too slow, but there may be chance for inaccuracy in fitband.

# ## Assumption and some improvement tips
# * the user seems doesnt have physical activity.
# * the person sits too long time. this makes an assumption he may be fat. BMI should be high.
# * the user should spent time on some physcial acitivity to improve his health.
# * I wish the user should be fit and lead a healthy life

# In[ ]:




