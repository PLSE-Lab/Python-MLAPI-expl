#!/usr/bin/env python
# coding: utf-8

# # Data Story 8 and 9

# This notebook is a part of my 8th day in analyzing datasets.
# 
# Let's jump straight to some Air Quality.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas_profiling import ProfileReport
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading in the data
data = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
data.head(2)


# In[ ]:


# creating a new year column
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year


# # Report of Air Quality Data in India

# In[ ]:


report = ProfileReport(data)
report


# # How AQI(air quality index) is distributed?

# In[ ]:


sns.set_style("darkgrid")
sns.kdeplot(data=data['AQI'],label="AQI" ,shade=True)


# # Which city has lowest AQI?

# In[ ]:


aqi = data.groupby('City')['AQI'].min().reset_index()
aqi  = aqi.sort_values("AQI")
aqi = aqi.head(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(aqi['AQI'].tolist(), labels=aqi['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()


# # Cities with lowest AQI per year

# In[ ]:


perc = data.loc[:,["year","City",'AQI']]
perc['mean_AQI'] = perc.groupby([perc.City,perc.year])['AQI'].transform('mean')
perc.drop('AQI', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Talcher','Amritsar','Brajrajnagar'] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values("year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_AQI", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()


# # Which City has highest AQI?

# In[ ]:


aqi = data.groupby('City')['AQI'].max().reset_index()
aqi  = aqi.sort_values("AQI")
aqi = aqi.tail(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(aqi['AQI'].tolist(), labels=aqi['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()


# # Cities with highest AQI per year

# In[ ]:


perc = data.loc[:,["year","City",'AQI']]
perc['mean_AQI'] = perc.groupby([perc.City,perc.year])['AQI'].transform('mean')
perc.drop('AQI', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Hyderabad','Amritsar','Gurugram','Guwahati',"Ahmedabad"] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values(by="year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_AQI", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()


# I will end my today's analysis here. Tomorrow I will continue with analysing the cities with highest AQI because they are the ones that need more attention and need to be improved.
# 
# I will also try to see if there was any decrease noticed in AQI that may provide insights about future steps to be taken.
# 
# See ya.

# # Looking at each of the cities with highest AQI

# In[ ]:


data.head()


# Let's first group the city columns and take the mean AQI values

# In[ ]:


data1 = data['AQI'].dropna()
top_10_city = data.loc[data1.index].groupby('City')['AQI'].mean().reset_index()
top_10_city.sort_values('AQI', ascending=False, inplace=True)
top_10_city.head(10)


# # Top 10 cities with highest AQI

# In[ ]:


top_cities = top_10_city.head(10)['City'].tolist()
top_cities


# Now there is one interesting thing to notice. Talcher is actually the city that once had the lowest AQI in the whole dataset in the span of 5 years. 
# 
# But when taken mean data it comes in one of the cities with highest AQI.
# 
# What this signifies is it's AQI value is increasing rapidly. Let's try to visualize its growth using a line plot over the years.

# # Talcher AQI map

# In[ ]:


talcher = data[data['City']=='Talcher']
data_by_year = talcher.groupby('year')['AQI'].mean().reset_index().dropna()
data_by_year.head()


# In[ ]:


plt.plot(data_by_year['year'], data_by_year['AQI'])
plt.xticks(data_by_year['year'].tolist())
plt.title('Year wise mean AQI for Talcher')
plt.xlabel('Years')
plt.ylabel('Mean AQI')
plt.show()


# The situation for talcher is not very good because it's AQI value saw a huge increase in 2020

# # Brajrajnagar AQI map

# In[ ]:


braj = data[data['City']=='Brajrajnagar']
data_by_year = braj.groupby('year')['AQI'].mean().reset_index().dropna()
data_by_year.head()


# In[ ]:


plt.plot(data_by_year['year'], data_by_year['AQI'])
plt.xticks(data_by_year['year'].tolist())
plt.title('Year wise mean AQI for Talcher')
plt.xlabel('Years')
plt.ylabel('Mean AQI')
plt.show()


# Brajrajnagar may have come in top 10 cities in case of the mean but it's showing  huge progress in the last couple of years

# Checking for all cities

# In[ ]:


fig = plt.figure(figsize=(15,28))
for city,num in zip(top_cities, range(1,11)):
    df = data[data['City']==city]
    data_by_year = df.groupby('year')['AQI'].mean().reset_index().dropna()
    ax = fig.add_subplot(5,2,num)
    ax.plot(data_by_year['year'], data_by_year['AQI'])
    ax.set_xticks(data_by_year['year'].tolist())
    ax.set_title('Year wise mean AQI for {}'.format(city))
    ax.set_ylabel('Mean AQI')


# Things aren't looking good for Kolkata, Talcher and Guwahati. All other cities are improving their AQI.

# I have invested enough time into understanding the AQI for each city.
# 
# Now let's move on to check the variation of the other variables(toxic factors)

# # Which city has highest NO?

# In[ ]:


data.head(2)


# Now as we can see there are a lot of null values.
# 
# One logical thing to do is to replace the NaN with 0.0 i.e, their value was not detected in that particular city.

# In[ ]:


#imputing null values with 0.0
df = data.fillna(0.0)


# In[ ]:


no = df.groupby('City')['NO'].mean().reset_index()
no  = no.sort_values("NO")
no = no.head(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(no['NO'].tolist(), labels=no['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()


# In[ ]:


perc = df.loc[:,["year","City",'NO']]
perc['mean_NO'] = perc.groupby([perc.City,perc.year])['NO'].transform('mean')
perc.drop('NO', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Hyderabad','Bhopal','Jorapokhar','Chennai',"Bengaluru"] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values("year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_NO", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()


# This visualization is actually not a good fit for checking all variables at once. 
# 
# Let's try with kdeplots for each country.

# # Distribution of NO, NO2 and more

# In[ ]:


fig = plt.figure(figsize=(15,23))
for city,num in zip(top_cities, range(1,11)):
    df = data[data['City']==city]
    df = df.groupby('year')['NO'].mean().reset_index().dropna()
    ax = fig.add_subplot(5,2,num)
    ax.set_title(city)
    sns.kdeplot(data=df['NO'],label="NO" ,shade=True)


# This shows the distribution but not the year wise data. Let's try a grouped bar plot.
# 
# I will try that tomorrow because I am out of time today

# Okay so while thinking on other accounts it's actually not much information can be extracted from just the increase or decrease of NO, NO2 etc.
# 
# So rather than visualize that. I will try to check its relation with AQI values.

# In[ ]:


data.head(2)


# In[ ]:


sns.scatterplot('PM2.5', 'AQI', hue='year', data=data)
plt.title('Relation between PM2.5 and AQI')
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.show()


# Now this seems interesting. Let's plot it for each of the variables.
# 
# First the variables need to be separated.

# In[ ]:


variables = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']

fig = plt.figure(figsize=(16,34))
for variable,num in zip(variables, range(1,len(variables)+1)):
    ax = fig.add_subplot(6,2,num)
    sns.scatterplot(variable, 'AQI', hue='year', data=data)
    plt.title('Relation between {} and AQI'.format(variable))
    plt.xlabel(variable)
    plt.ylabel('AQI')


# Now let's check te relations one by one for each of the variables.

# 1. PM2.5 -> This shows a inversely proportional behaviour since when it's value remains low the AQI value increases while in other cases when its value increases the AQI value reamins low.
# 
# 2. PM10 -> This also shows an inverse behaviour except for the fact that even when PM10 value is less AQI value doesn't increase much
# 
# Similar analysis can be done for each of the other variables.

# In[ ]:


cities = ['Delhi', 'Ahmedabad', 'Hyderabad', 'Bengaluru', 'Kolkata']
fig,ax = plt.subplots(figsize=(15, 7))

for city in cities: 
    sns.lineplot(x="Date", y="AQI", data=data[data['City']==city].iloc[::30],label = city)

ax.set_xticklabels(ax.get_xticklabels(cities), rotation=30, ha="left")

ax.set_title('AQI values in cities')
ax.legend()


# I will analysis here. because i am running out of more ideas to analyze.
# 
# Will come back to this if there is any more new face added to the data.

# See you in the next data story. Bye

# In[ ]:




