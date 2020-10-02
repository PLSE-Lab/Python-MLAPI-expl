#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from mpl_toolkits.basemap import Basemap
from IPython.core.display import display, HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Introduction

# In this Python Notebook, I will try to answer some of the questions posed by the dataset owner. This is my initial work on the dataset and I'm not taking into consideration some issues in the dataset, such as duplications. I will treat this in a second run of this exploratory data analysis together with the remaining questions.

# In[ ]:


df = pd.read_csv('../input/Mass Shootings Dataset.csv', encoding = 'ISO-8859-1', parse_dates=['Date'])
df.drop(['S#'], axis=1, inplace=True)
df.head()


# There is one NaN value in the Gender column, which corresponds to Las Vegas' mass shooter Stephen Paddock. Therefore, I'll fill that value with 'Male'

# In[ ]:


df['Gender'].fillna('Male', inplace=True)


# In[ ]:


print(df['Gender'].unique())


# In[ ]:


df.Gender.replace(['M', 'F', 'M/F'], ['Male', 'Female', 'Male/Female'], inplace=True)


# In[ ]:


print(df['Gender'].unique())


# In[ ]:


df.groupby('Gender').count()


# In[ ]:


df['Year'] = df['Date'].dt.year


# ## How many people got killed and injured per year?

# In[ ]:


fatalities_year = df[['Fatalities', 'Year']].groupby('Year').sum()

fatalities_year.plot.bar(figsize=(12,6), color='red')
plt.ylabel('Fatalities', fontsize=12)
plt.title('Number of Fatalities per Year', fontsize=18)


# In[ ]:


fatalities_year


# The following years were the most violent in the 50 years of mass shooting, resulting in the demise of more than 100 people per year.

# In[ ]:


fatalities_year[fatalities_year['Fatalities'] > 100]


# In[ ]:


injured_year = df[['Injured', 'Year']].groupby('Year').sum()

injured_year.plot.bar(figsize=(12,6))
plt.ylabel('Injured', fontsize=12)
plt.title('Number of Injured per Year', fontsize=18)


# In[ ]:


injured_year


# In[ ]:


injured_year[injured_year['Injured'] > 100]


# How does the number of fatalities compare to the number of injuries? Are the number of injured people always high than killed ones?

# In[ ]:


tot_victims = df[['Year', 'Injured', 'Fatalities']].groupby('Year').sum()

tot_victims.plot.bar(figsize=(12,6))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Number of Fatalities vs Injuries per Year', fontsize=18)


# The below graph shows that in most cases the number of injured people is higher than the number of killed ones. We can see that 1989, 1998 and 2017 have an exceptionally high number of injured people. The below table summarises the total number of fatalities vs injures per year.

# In[ ]:


df[['Year','Fatalities', 'Injured', 'Total victims']].groupby('Year').sum()


# To summarise, in the last 50 years there were:

# In[ ]:


print('Total Fatalities: ' + str(df['Fatalities'].sum()))


# In[ ]:


print('Total Injured: ' + str(df['Injured'].sum()))


# In[ ]:


print('Total Number of Victims: ' + str(df['Total victims'].sum()))


# And, the graph below shows the number of attacks per year.

# In[ ]:


year_count = df['Year'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(year_count.index, year_count.values, alpha=0.8, color=color[2])
plt.xticks(rotation='vertical')
plt.xlabel('Year of Shooting', fontsize=12)
plt.ylabel('Number of Attacks', fontsize=12)
plt.title('Number of Attacks per Year', fontsize=18)
plt.show()


# ## Visualize mass shootings on US map

# In[ ]:


# U.S. center lat and long
center_lat = 39.8283
center_lon = -98.5795

df_positions = df[['Latitude', 'Longitude', 'Total victims']].dropna()


# In[ ]:


plt.figure(figsize=(16,8))

latitudes = np.array(df_positions['Latitude'])
longitudes = np.array(df_positions['Longitude'])

lons, lats = np.meshgrid(longitudes,latitudes)

m = Basemap(projection='mill',llcrnrlat=20,urcrnrlat=50,                llcrnrlon=-130,urcrnrlon=-60,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.fillcontinents(color='#04BAE3', lake_color='#FFFFFF')
m.drawmapboundary(fill_color='#FFFFFF')

x, y = m(longitudes, latitudes)

m.plot(x, y, 'ro')

plt.title("Mass Shooting Attacks on US Map")


# ## Is there any correlation between shooter and his/her race, gender?

# First, let's analyse the gender distribution of the perpetrators.

# In[ ]:


gender = df['Gender'].value_counts()
gender


# In[ ]:


gender.sort_values().plot(kind='bar', figsize=(12,6), fontsize=12).set_title(
    'Gender Distribution of Mass Shootings')


# In[ ]:


gender_prop = pd.Series()

for key, value in gender.iteritems():
    gender_prop[key] =  gender[key] / len(df) * 100
    
gender_prop


# As we can see in this analysis, there were a total of 398 mass shootings in the USA in the last 50 years. The great majority of perpetrators were male, who commited 365 of the shootings, i.e., 91.5% of the cases. The remainder is: Female: 1.8%, Male/Female: 1.3%, Unknown: 5.5%.
# The below table shows the total number of victims by perpetrators gender.

# In[ ]:


df[['Gender', 'Total victims']].groupby('Gender').sum()


# As expected, most of the victims (4119 people) suffered their ordeal or their demise in the hands of male perpetrators.
# 
# Now, let's do our analysis by race.

# In[ ]:


df['Race'].value_counts()


# Let's consolidate the races, e.g., "White American or European American", "white", "White" will all be considered "white" and so on.

# In[ ]:


df.loc[['white' in str(x).lower().split() for x in df['Race']], 'Race']= 'white'
df.loc[['black' in str(x).lower().split() for x in df['Race']], 'Race']= 'black'
df.loc[['asian' in str(x).lower().split() for x in df['Race']], 'Race']= 'asian'
df.loc[['native' in str(x).lower().split() for x in df['Race']], 'Race']= 'native american'

race_counts = df['Race'].value_counts()
race_counts


# In[ ]:


race_counts.sort_values().plot(kind='bar', figsize=(12,6), fontsize=12).set_title('Race distribution of mass shootings')


# In[ ]:


race_prop = pd.Series()

for key, value in race_counts.iteritems():
    race_prop[key] = (race_counts[key] / len(df)) * 100
    
race_prop


# ## Any correlation with calendar dates? Do we have more deadly days, weeks or months on average?

# In the section, we will analyse the correlation of the shootings with calendar date. We will start by analysing the number of attacks and victims on a monthly, weekly and daily basis. Then, at the end we'll try to find any correlation.

# In[ ]:


df['Month'] = df['Date'].dt.month
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])


# In[ ]:


month = df['Month'].value_counts()
month.sort_index().plot.bar(figsize=(12,6), alpha=0.8, color=color[1])
plt.ylabel('Number of Shooting', fontsize=12)
plt.title('Number of Shootings per Month', fontsize=18)


# The above graph shows that February is the month with the greatest amount of shootings (63), followed by March with 52 shootings. In the next graph, we will identify the month is the greatest number of victims.

# In[ ]:


month_df = df[['Month', 'Total victims']].groupby('Month').sum()
month_df.plot.bar(figsize=(12,6), alpha=0.8, color=color[3])
plt.ylabel('Total Number of Victims', fontsize=12)
plt.title('Number of Victims per Month', fontsize=18)


# As we can see on the above graph and corresponding table, October seems to be the most violent month. However, it is important to notice that this is the case due to the 573 victims of the Las Vegas Mass Shooting at the beginning of October 2017. Having that in mind, we could consider February, April, July and December to be quite violent.

# Next, we'll analyse the number of shooting per week and per day....

# ## How many shooters have some kind of mental health problem?

# In[ ]:


df.loc[['unknown' in str(x).lower().split() for x in df['Mental Health Issues']], 'Mental Health Issues']= 'Unknown'
df.loc[['unclear' in str(x).lower().split() for x in df['Mental Health Issues']], 'Mental Health Issues']= 'Unclear'

mental_issues = df['Mental Health Issues'].value_counts()

print(mental_issues)


# In[ ]:


df.set_index('Date', inplace=True)
df.sort_index(inplace=True)


# In[ ]:


ax = df[df['Mental Health Issues'] == 'Yes']['Total victims'].plot(style='o', label='Yes', figsize=(14,10))
df[df['Mental Health Issues'] == 'No']['Total victims'].plot(style='o', label='No', ax=ax)
df[df['Mental Health Issues'] == 'Unknown']['Total victims'].plot(style='o', label='Unknown', ax=ax)
plt.title('Mental Health Issues')
ax.legend()


# In[ ]:




