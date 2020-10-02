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

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_ctemp = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
df_ctemp.head()


# In[ ]:


df_ctemp.shape


# In[ ]:


df_ctemp['Year'].unique()


# Looks like there are some rows with invalid values in the Year column. Lets remove them.

# In[ ]:


df_ctemp = df_ctemp[~df_ctemp['Year'].isin(['201','200'])] 
df_ctemp.shape


# In[ ]:


df_ctemp['Month'].unique()


# Month data looks good, lets check the Day column.

# In[ ]:


df_ctemp['Day'].unique()


# Lets remove the rows with invalid value 0 in Day column from the dataset.

# In[ ]:


df_ctemp = df_ctemp[df_ctemp['Day'] != 0] 
df_ctemp.shape


# Lets create the Date column from available Month, Day and Year columns as Date columns comes in handy with analysis.

# In[ ]:


df_ctemp['Date'] = df_ctemp['Year'].astype(str) + '-' + df_ctemp['Month'].astype(str) + '-' + df_ctemp['Day'].astype(str)
df_ctemp.head()


# Now lets convert the Date column to the datetime format for it to be actually useful.

# In[ ]:


df_ctemp['Date'] = pd.to_datetime(df_ctemp['Date'])
df_ctemp.info()


# Lets check if we have any NAN values.

# In[ ]:


df_ctemp.isna().sum()


# Lets do a deep dive to understand these NAN values in State column.

# In[ ]:


df_ctemp[['Region','Country','State','City','Month','Day','Year','AvgTemperature']].loc[(df_ctemp['State'].notnull())]


# In[ ]:


df_ctemp['State'].unique()


# Looks like State data is available only for US, which is fine but lets convert the remaining NANs in this column to NA to make it logical.

# In[ ]:


df_ctemp['State'].fillna('NA', inplace=True)


# In[ ]:


df_ctemp.isna().sum()


# In[ ]:


df_ctemp.describe()


# Looking at the temperature data, we know it is in Fahrenheit and will convert it to Celcius for better understanding. 
# 
# But before we do that we need to take care of the invalid data -99 F which is definitely a data issue. Instead of removing it, lets convert to NAN and replace by forward fill.

# In[ ]:


df_ctemp = df_ctemp.replace([-99.00], np.nan)
df_ctemp.isna().sum()


# In[ ]:


df_ctemp['AvgTemperature'] = df_ctemp['AvgTemperature'].fillna(method = 'ffill')
df_ctemp.isna().sum()


# In[ ]:


df_ctemp.shape


# Now lets see the temperature in Celcius.

# In[ ]:


df_ctemp['AvgTemperature'] = round((((df_ctemp['AvgTemperature'] - 32) * 5) / 9),2)
df_ctemp.head()


# Lets check the spread once again to see if we have any other data issues.

# In[ ]:


df_ctemp.describe()


# Dataset looks good now. Lets visualize Avg. Temperature across all cities and years.

# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(15, 6))
bins = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

sns.distplot(df_ctemp['AvgTemperature'], bins= bins, color="steelblue")

mean_temp = np.mean(df_ctemp['AvgTemperature'])
plt.axvline(mean_temp, label= 'Mean Avg. Temp.', color= 'green')

median_temp = np.median(df_ctemp['AvgTemperature'])
plt.axvline(median_temp, label= 'Median Avg. Temp.', color= 'red')

plt.legend()
plt.title('Global Avg. Temp. Distribution')
plt.xlabel('Avg. Temp. (in Celcius)')
plt.xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], 
           ['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'],
           rotation=20)
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:


print('Skewness: ', df_ctemp['AvgTemperature'].skew())
print('Kurtosis: ', df_ctemp['AvgTemperature'].kurtosis())


# As we can see from the histogram as well as the Skewness value, the dataset is negatively skewed moderately. Also since the dataset is platykurtic, we don't expect to see a lot of outliers. However, the situation might change if we look at the region or country level.
# 
# Most of the temperatures across the world and time of the year is concentrated between 20 and 30 C.
# 
# Lets now see how is the mean average temperature is changing over time.

# In[ ]:


world_temp = pd.Series(round(df_ctemp.groupby('Date')['AvgTemperature'].mean().sort_values(),2))

sns.set_style("darkgrid")
plt.figure(figsize=(18, 6))

sns.lineplot(data= world_temp, color= 'blue')
plt.xlabel('Time')

plt.ylabel('Temperature (in Celcius)')
plt.title('World Mean Avg. Temperature Over Time')
plt.show()


# The mean highest temperature has more or less remained fixed at around 25 C while the mean lowest temperatures did vary meaning the winters are getting warmer.
# 
# Though the above visualization is interesting, we cannot interpret anything meaningful further because the world is geographically diverse and the temperatures will surely vary across geographies and time of the year.
# 
# Perhaps, we should zoom in to regions to get a better sense of what is going on.

# In[ ]:


region_temp = pd.Series(round(df_ctemp.groupby('Region')['AvgTemperature'].mean().sort_values(),2))
#Select the style of the plot
style.use('ggplot')
region_temp.plot(kind='barh',
                 figsize=(10, 5),
                 color='chocolate',
                 alpha=0.75)
plt.xlabel('Mean Avg. Temperature')
plt.ylabel('Region')
plt.title('Mean Avg. Temperature By Region')
plt.show()


# No surprises here, irrespective of the time of the year the mean average temperature of Africa is a sharp contrast to Europe whereas Asis is somewhere in between.
# 
# Lets visualize other parameters for these regions as well.

# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
sns.boxplot(x= df_ctemp['Region'], y= df_ctemp['AvgTemperature'])
plt.xlabel('Region')
plt.xticks(rotation= 20)
plt.ylabel('Temperature (in Celcius) Spread')
plt.title('World Temperature (in Celcius) Spread')
plt.show()


# There seems to be a lot of outliers especially on the minimum temperature side for all the regions. This could be because of the -99 F we treated above.

# In[ ]:


df_asia = df_ctemp[df_ctemp['Region'] == 'Asia']
df_asia


# In[ ]:


asia_temp = pd.Series(round(df_asia.groupby('Date')['AvgTemperature'].mean().sort_values(),2))

sns.set_style("darkgrid")
plt.figure(figsize=(18, 6))

sns.lineplot(data= world_temp, color= 'blue')
plt.xlabel('Time')

plt.ylabel('Temperature (in Celcius)')
plt.title('Asia Mean Avg. Temperature Over Time')
plt.show()


# Looks like Asia follows the overall world trend, where the winters are becoming warmer whereas the summer is more or less the same.
# 
# Lets deep dive into India.

# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
sns.boxplot(x= df_asia['Country'], y= df_ctemp['AvgTemperature'])
plt.xlabel('Country')
plt.xticks(rotation= 20)
plt.ylabel('Temperature (in Celcius) Spread')
plt.title('Asia Temperature (in Celcius) Spread')
plt.show()


# In[ ]:


df_india = df_ctemp[df_ctemp['Country'] == 'India']
df_india


# In[ ]:


df_india['City'].unique()


# Lets visualize how temperature is varying in the 4 Indian cities in the dataset over time.

# In[ ]:


g = sns.FacetGrid(df_india, col= 'City', col_wrap= 2, palette= "Set3", height= 4, aspect= 3, margin_titles=True)
g.map(sns.pointplot,'Year','AvgTemperature')
g.set(yticks= [20, 25, 30, 35])


# Over time, the deviation is not much for any of the 4 cities.
# 
# Now lets do a comparitive study of the temperatures across the 4 cities.

# In[ ]:


g = sns.FacetGrid(df_india, col= 'City')
g.map(sns.distplot, 'AvgTemperature', rug=False)
g.add_legend()


# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
sns.boxplot(x= df_india['City'], y= df_ctemp['AvgTemperature'])
plt.xlabel('City')
plt.xticks(rotation= 20)
plt.ylabel('Temperature (in Celcius) Spread')
plt.title('Indian Cities Temperature (in Celcius) Spread')
plt.show()


# Unlike the other 3 cities, Delhi has a very hot summer and a very cold winter -- something which is clearly visible in the above graphs.
# 
# Now lets deep dive onto Delhi as it has bigger spread of data.

# In[ ]:


df_delhi = df_india[df_india['City'] == 'Delhi']
df_delhi_winter = df_india[df_india['Month'] == 1]
df_delhi_winter = df_delhi_winter[['Month','Day','Year','AvgTemperature']]
df_delhi_winter


# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(20, 6))
df_delhi_winter.groupby(['Year','Month'])['AvgTemperature'].mean().plot()
plt.xlabel('Year-Month')
plt.ylabel('Temperature (in Celcius)')
plt.title('Delhi Temperature Over Time')
plt.show()


# In[ ]:


df_delhi_summer = df_india[df_india['Month'] == 5]
df_delhi_summer = df_delhi_summer[['Month','Day','Year','AvgTemperature']]
df_delhi_summer


# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(20, 6))
df_delhi_summer.groupby(['Year','Month'])['AvgTemperature'].mean().plot()
plt.xlabel('Year-Month')
plt.ylabel('Temperature (in Celcius)')
plt.title('Delhi Temperature Over Time')
plt.show()


# As we can see, for a sample month of January which is usually the coldest month in Delhi, the mean temperature has slowly risen over the years. Similarly, the summers have become hotter as well. Sure sign of global warming!
