#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
get_ipython().run_line_magic('pylab', 'inline')
p = 'YlOrRd'


# In[ ]:


df = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.drop(["DR Number", # For search purpose not useful here
         "Area ID", # We have Area Name
         "Crime Code", # Uniform
         "Crime Code Description", # Uniform
         "Premise Code",  # Could use Premise Description
         "Date Reported", # More intrested in date occured
         "Neighborhood Councils (Certified)", # Meaningless without description
         "Census Tracts", # Meaningless without description
         "Council Districts", # Irrelevant
         "MO Codes", # Too much null values
         "LA Specific Plans"], # Too much null values
         axis=1, inplace=True)


# In[ ]:


pd.isnull(df).sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.columns = ['date', 'hour', 'area', 'district', 'victim_age', 'victim_sex', 'victim_race', 'premise', 'address', 'cross_street', 'location', 'zip_code', 'precinct_boundary']


# In[ ]:


race_dict = {'H':'Hispanic', 'B':'Black', 'O':'Unknown', 'W':'White', 'X':'Unknown', '-':'Unknown',
             'A':'Asian', 'K':'Asian', 'C':'Asian', 'F':'Asian', 'U':'Pacific Islander',
             'J':'Asian', 'P':'Pacific Islander', 'V':'Asian', 'Z':'Asian',
             'I':'American Indian', 'G':'Pacific Islander', 'S':'Pacific Islander', 'D':'Asian', 'L':'Asian'}
df.victim_race = df.victim_race.map(race_dict)


# In[ ]:


df.victim_race.value_counts()


# In[ ]:


df.date = pd.to_datetime(df.date)
df["year"] = df.date.dt.year
df["day_of_week"] = df.date.dt.dayofweek
df.hour = df.hour.astype(str)
df.hour = [i[:2] if len(i) == 4 else i[0] for i in df["hour"]]
df.hour = df.hour.astype(int)


# In[ ]:


df['location'] = [ast.literal_eval(d) for d in df.location]
df['longitude'] = [d['longitude'] for d in df.location]
df['latitude'] = [d['latitude'] for d in df.location]
df.longitude = df.longitude.astype(float)
df.latitude = df.latitude.astype(float)


# In[ ]:


df.index = pd.DatetimeIndex(df.date)


# In[ ]:


df.head()


# ## Total accidents per day

# In[ ]:


# mean and standard deviation of accidents per day
accidents_per_day = pd.DataFrame(df.resample('D').size())
accidents_per_day['mean'] = df.resample('D').size().mean()
accidents_per_day['std'] = df.resample('D').size().std()
# upper control limit and lower control limit
UCL = accidents_per_day['mean'] + 3 * accidents_per_day['std']
LCL = accidents_per_day['mean'] - 3 * accidents_per_day['std']
plt.figure(figsize=(15,6))
df.resample('D').size().plot(label='Accidents per day', color='sandybrown')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
accidents_per_day['mean'].plot(color='red', linewidth=2, label='Average')
plt.title('Total accidents per day', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of accidents')
plt.tick_params(labelsize=14)


# In[ ]:


# we notice the last month's records are incomplete, so let's get rid of it
date_before = datetime.date(2019, 6, 1)
df = df[df.date < date_before]


# ## Total accidents per month

# In[ ]:


month_df = df.resample('M').size()
plt.figure(figsize=(15,6))
month_df.plot(label='Total,  accidents per month', color='sandybrown')
month_df.rolling(window=12).mean().plot(color='red', linewidth=5, label='12-Months Average')
plt.title('Total accidents per month', fontsize=16)


# In[ ]:


print("Best Month {0}: {1}".format(month_df.idxmin(), month_df[month_df.idxmin()]))
print("Worst Month {0}: {1}".format(month_df.idxmax(), month_df[month_df.idxmax()]))


# > - There's a trend of increasing accidents per month
# > - Feburary 2013 had the least amount of accidents
# > - December 2018 was the worst with 4201 accidents

# In[ ]:


weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
week_df = pd.DataFrame(df["day_of_week"].value_counts()).sort_index()
week_df["day"] = weekdays
week_df.columns = ["Accident counts", "Week day"]
week_df


# ## Accidents per weekday

# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(x="Week day", y="Accident counts", color="coral", data=week_df)


# ## Average number of accidents per race and hour

# In[ ]:


accidents_hour_pt = df.pivot_table(index='victim_race', columns='hour', aggfunc='size')
accidents_hour_pt = accidents_hour_pt.apply(lambda x: x / accidents_hour_pt.max(axis=1))
plt.figure(figsize=(15,5))
plt.title('Average Number of Accidents per Race and Hour', fontsize=14)
sns.heatmap(accidents_hour_pt, cmap=p, cbar=True, annot=False, fmt=".0f");


# ## Moving average of accidents per month by race

# In[ ]:


race_df = df.pivot_table(index='date', columns='victim_race', aggfunc='size', fill_value=0).resample('M').sum()
race_df.drop(["Unknown"], axis=1, inplace=True)
race_df.rolling(window=12).mean().plot(figsize=(15,6), linewidth=4, cmap=p)
plt.title('Moving average of accidents per month by race', fontsize=16)
plt.xlabel('Year')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# ## Moving average of accidents per month by race (ratio)

# ### LA Population Composition by race
# > - White (non-hispanic): 28.5%
# > - Black: 9.0%
# > - American Indian: 0.4%
# > - Asian: 11.6%
# > - Pacific Islanders: 0.2%
# > - Hispanic: 48.2%
# > - Other: 0.6%  
# src: https://statisticalatlas.com/place/California/Los-Angeles/Race-and-Ethnicity & U.S. Census Bureau. American Community Survey, 2011 American Community Survey 5-Year Estimates, Table B02001. American FactFinder Archived September 11, 2013, at the Wayback Machine. Retrieved October 26, 2013.

# In[ ]:


race_df["White"] = race_df["White"]/0.285
race_df["Black"] = race_df["Black"]/0.09
race_df["American Indian"] = race_df["American Indian"]/0.004
race_df["Asian"] = race_df["Asian"]/0.116
race_df["Pacific Islander"] = race_df["Pacific Islander"]/0.002
race_df["Hispanic"] = race_df["Hispanic"]/0.482


# In[ ]:


race_df.rolling(window=12).mean().plot(figsize=(15,6), linewidth=4, cmap=p)
plt.title('Moving average of accidents per month by race (ratio)', fontsize=16)
plt.xlabel('Year')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# ## Victim age distribution

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(df.victim_age, color='coral')


# ## 2010 and 2018 geoplot comparison

# In[ ]:


df_2010 = df[df.year==2010]
df_2018 = df[df.year==2018]

print("2010 monthly average: {}".format(df_2010.resample('M').size().mean()))
print("2018 monthly average: {}".format(df_2018.resample('M').size().mean()))


# In[ ]:


plt.figure(figsize=(15,20))
sns.scatterplot(x='longitude', y='latitude', hue='victim_race', palette=p, data=df_2010.sample(3014))
plt.title("Average month in 2010")


# In[ ]:


plt.figure(figsize=(15,20))
sns.scatterplot(x='longitude', y='latitude', hue='victim_race', palette=p, data=df_2018.sample(3817))
plt.title("Average month in 2018")

