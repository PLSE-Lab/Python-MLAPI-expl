#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's try to open all the data and concated it into one dataframe

# In[ ]:


df_list = []
for i in os.listdir("../input"):
    df = pd.read_csv('../input/'+i, header = 0, index_col = None)
    df_list.append(df)

bikes = pd.concat(df_list, axis = 0, ignore_index = True)    


# In[ ]:


bikes.head()


# Because the dataset are separated for each month, lets see how many row initially for each month

# In[ ]:


for i in bikes['month'].unique():
    print('Month {}: {}'.format(i, len(bikes[bikes['month'] == i])))


# In[ ]:


bikes.info()


# Seems there are missing values in the data. Lets see where the data is missing

# In[ ]:


bikes.isnull().any()


# In[ ]:


bikes[pd.isna(bikes).any(axis = 1)]


# In[ ]:


print('Missing value in each month:')
for i in bikes['month'].unique():
    print('Month {}: {}'.format(i,len(bikes[(bikes['month'] == i) & (pd.isna(bikes).any(axis = 1))])))
    


# There are 50111 row with missing value but scattered in multiple columns. The missing data could be contributed by the user who lost the bike, or just did not want their age and/or gender being shown. 
# 
# Seems for each month it is only around 5% of the data is missing, I assume it was safe to just drop the data.

# In[ ]:


bikes.dropna(inplace = True)


# In[ ]:


bikes.info()


# ## Data Exploration
# Time to explore the dataset. Lets see what kind of insight we could get

# In[ ]:


bikes.describe()


# Just by looking at the birth year,the furthest were born at 1878! This user either input the birth year wrong or just really healthy to keep biking on that age. Either way, lets see the birth year column more closely.

# In[ ]:


plt.figure(figsize = (12, 8))
sns.set_style('darkgrid')
sns.distplot(bikes['member_birth_year'])


# Seems there are some user that born as far as in 1800s. I do not want to made assumption that they input the birth year wrong, as it still possible to be happen. For that, I would grouping the user to certain age group categories (Children (00-14 years), Youth (15-24 years), Adults (25-64 years), Seniors (65 years and over)). Why this age group? There are many age group that we could made, but for this case I am just gonna refer to the site I found about the age group. You could check it here: https://www.statcan.gc.ca/eng/concepts/definitions/age2
# 
# I am working on this dataset in 2019, so Children born in 2005 - 2019, Youth born in 1995 - 2004, Adults born in 1955 - 1994, and Seniors born before 1955. 

# In[ ]:


def group_birth(year):
    if year >= 2005 and year <= 2019 :
        return 'Children'
    elif year >= 1995 and year <= 2004:
        return 'Youth'
    elif year >= 1955 and year <= 1994:
        return 'Adult'
    else:
        return 'Senior'
    
bikes['age_group'] = bikes['member_birth_year'].apply(group_birth)        


# In[ ]:


sns.countplot(bikes['age_group'])


# Most of the transaction were made by the adult, and no Children were present at all. I could assume that baywheel have requirement that you have to be at certain age to be allowed borrowing the bike, but as I have not yet reading anything about the policy I would stay away from this assumption for now. Either way, the data shown that there are no Children at all use the baywheel services in January 2019 - May 2019; with Adult dominate the user age group. This is understable, as Adult age group have the widest year distribution. 

# In[ ]:


print('Number of Services by:')
for i in bikes['age_group'].unique():
    print('{} used {} times for baywheel services'.format(i, len(bikes[bikes['age_group'] == i])))


# As an additional insight, now I want to see the differences between user type and the age group. Just to see if there is an interesting pattern

# In[ ]:


sns.countplot(data = bikes, x= 'age_group', hue = 'user_type')


# In[ ]:


len(bikes[(bikes['age_group'] == 'Senior') & (bikes['user_type'] == 'Customer')])


# Most of the user are Subscriber to the baywheels, and many of them are Adults. I would assume that many of the customer are just one-time user, or the user have been converted to the Subscriber. With this assumption, the baywheel marketing strategy seems work well.
# 
# This time, I would try to plot every numerical feature with each other. Just to see if there are interesting insight to be seen.

# In[ ]:


sns.pairplot(bikes, hue = 'month')


# Not much insight could be seen when separated by the month. The most interesting thing that could be seen here is that certain bike is used more in certain month contrary to the other. Lets see from another group.

# In[ ]:


sns.pairplot(bikes, hue = 'age_group')


# There are 2 things that was interesting from the plot here. First, almost all senior only use the bike service for a short trip; I would come back to this plot later to see the pattern more closely. Second, there is interesting pattern in the start and end station id. We could see there are station with the youth mostly use the service in those area. I would also gonna look at this pattern more closely. 
# 
# I would try to check the trip_duration_sec distribution first, as from the data seems spreading so far.

# In[ ]:


plt.figure(figsize = (12,8))
sns.distplot(bikes['trip_duration_sec'])


# Seems the data is skew so far. In this case, I would try to remove the outlier rather than keep the outlier. This outlier could give an interesting sight, but right now I am would just keep the data that was in the normal distribution as these data is the representation of the most common cases.
# 
# I would standarize the data and I would use the 68-95-99.7 rule to remove the outlier. It means, I would remove the standarize data that were higher than 3 or lesser than -3. 

# In[ ]:


from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
bikes['trip_duration_sec_standard'] = standard.fit_transform(np.array(bikes['trip_duration_sec']).reshape(-1,1))
bikes = bikes[(bikes['trip_duration_sec_standard'] > -3) & (bikes['trip_duration_sec_standard'] < 3)]


# In[ ]:


len(bikes)


# After eliminate all the outlier, lets try to create the plot to see the current distribution now. As I am only interested now to see the overall pattern, and not the actual number itself. I would use the KDE plot.

# In[ ]:


plt.figure(figsize = (12,8))
sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['trip_duration_sec'], color = 'darkred', label ='Youth')
sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['trip_duration_sec'], color = 'blue', label ='Adult')
sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['trip_duration_sec'], color = 'green', label ='Senior')
plt.legend(frameon = False)


# Now we can see the distribution more clearly. My assumption is wrong, It seems all kind of age group are mostly using the the bike service only for short time (under 15 min). I have not yet check the start and end station point of the consumer, as it would give a better insight of the bikes usage. 
# 
# Speaking of the station, lets try to plot the start station and end station

# In[ ]:


plt.figure(figsize = (12,8))
sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['start_station_id'], color = 'darkred', label ='Youth')
sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['start_station_id'], color = 'blue', label ='Adult')
sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['start_station_id'], color = 'green', label ='Senior')
plt.title('Start Station Based on the Age Group')
plt.legend(frameon = False)


# In[ ]:


plt.figure(figsize = (12,8))
sns.kdeplot(bikes[bikes['age_group'] == 'Youth']['end_station_id'], color = 'darkred', label ='Youth')
sns.kdeplot(bikes[bikes['age_group'] == 'Adult']['end_station_id'], color = 'blue', label ='Adult')
sns.kdeplot(bikes[bikes['age_group'] == 'Senior']['end_station_id'], color = 'green', label ='Senior')
plt.title('End Station Based on the Age Group')
plt.legend(frameon = False)


# There seems a similar pattern for the start and end station id, which imply that almost all the usage of the bike either start and stop at the same station or most of the user follow the same route most of the time (If the start station and end route is similar from and back to). This need further examination of the route, but lets leave it for later.
# 
# The second interesting pattern is that there is a massive peak for Youth compared to the other age group. Although all the group follow almost the similar pattern, but we could still see the difference in the peak. This could imply that there are more Youth using bikes in certain areas compared to the other. I would also further examine this pattern later.

# In[ ]:


sns.pairplot(bikes, hue='user_type')


# 

# I would continue the exploration later after I have finish my own work. There are still many insight could be acquired from this data. 
