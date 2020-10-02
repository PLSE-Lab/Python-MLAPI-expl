#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
import cufflinks as cf
import plotly.tools as tls
import plotly.plotly as py
import folium
init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
p = sns.cubehelix_palette(15)
p2 = sns.cubehelix_palette(10)
p3 = sns.cubehelix_palette(24)
p4 = sns.cubehelix_palette(5)


# In[ ]:


df = pd.read_csv('../input/crime.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Process Date

# In[ ]:


df.FIRST_OCCURRENCE_DATE = pd.to_datetime(df.FIRST_OCCURRENCE_DATE)
df["YEAR"] = df.FIRST_OCCURRENCE_DATE.dt.year
df["DAY"] = df.FIRST_OCCURRENCE_DATE.dt.day
df["DAY_OF_WEEK"] = df.FIRST_OCCURRENCE_DATE.dt.dayofweek
df["MONTH"] = df.FIRST_OCCURRENCE_DATE.dt.month
df["HOUR"] = df.FIRST_OCCURRENCE_DATE.dt.hour
df.index = pd.DatetimeIndex(df["FIRST_OCCURRENCE_DATE"])


# In[ ]:


df.head()


# In[ ]:


print("This dataset ranges from {} to {}".format(df.index.min(), df.index.max()))


# In[ ]:


# The last month's data is incomplete, so let's get rid of it
date_before = pd.Timestamp(2019, 7, 1)
df = df[df.FIRST_OCCURRENCE_DATE < date_before]


# ## Crime vs Traffic Accidents

# In[ ]:


plt.figure(figsize=(6,5))
crime_and_traffic = pd.crosstab(index=df['IS_CRIME'],
                                columns=df['IS_TRAFFIC'])
crime_and_traffic.index = ["not crime", "crime"]
crime_and_traffic.columns = ["not traffic", "traffic"]
sns.heatmap(crime_and_traffic, annot=True, fmt="d", cmap=p)


# ## Offense categories distribution

# In[ ]:


plt.figure(figsize=(8,8))
cat_freq = df.OFFENSE_CATEGORY_ID.value_counts()
sns.countplot(y="OFFENSE_CATEGORY_ID", data=df, order=cat_freq.index, palette=p)


# > - Excluding traffic accidents the most common category is public disorder
# > - Then comes larceny (theft of personal items)
# > - We might need to do more analysis with all-other-crimes
# > - Murder is very rare in this dataset

# ## Most and least common offense types excluding traffic accidents

# In[ ]:


f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
type_freq = df.OFFENSE_TYPE_ID.value_counts()
common_types = type_freq.iloc[1:11]
rare_types = type_freq.iloc[-10:]
axes[0].set_title("Most common offense types")
sns.countplot(y="OFFENSE_TYPE_ID", data=df, order=common_types.index, palette=p2, ax=axes[0])
axes[1].set_title("Least common offense types")
sns.countplot(y="OFFENSE_TYPE_ID", data=df, order=rare_types.index, palette=p2, ax=axes[1])


# > - A large number of offenses are vehicle-related
# > - Public disorder is a common category but riots are actually rare (rare enough it's actually a riot instead of riots)
# > - Can't believe theft of cable services is a thing :D

# ## All other crimes

# In[ ]:


other_crimes = df[df.OFFENSE_CATEGORY_ID == "all-other-crimes"]
other_crimes_freq = other_crimes.OFFENSE_TYPE_ID.value_counts()
f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
other_common_types = other_crimes_freq.iloc[1:11]
other_rare_types = other_crimes_freq.iloc[-10:]
f.suptitle("All other crimes", fontsize=32)
axes[0].set_title("Most common offenses")
sns.countplot(y="OFFENSE_TYPE_ID", data=other_crimes, order=other_common_types.index, palette=p2, ax=axes[0])
axes[1].set_title("Least common offenses")
sns.countplot(y="OFFENSE_TYPE_ID", data=other_crimes, order=other_rare_types.index, palette=p2, ax=axes[1])


# ## Distribution of crime vs traffic over months

# In[ ]:


crimes_df = df[df.IS_CRIME==1]
traffic_df = df[df.IS_TRAFFIC==1]


# In[ ]:


f, axes = plt.subplots(1,2)
f.set_figheight(6)
f.set_figwidth(13)
plt.subplots_adjust(wspace=.5)
axes[0].set_title("Crime")
sns.countplot(x="MONTH", data=crimes_df, palette=p, ax=axes[0])
axes[1].set_title("Traffic Accidents")
sns.countplot(x="MONTH", data=traffic_df, palette=p, ax=axes[1])


# ## Distribution of crime vs traffic per hour

# In[ ]:


sns.countplot(x="HOUR", data=crimes_df, palette=p3)


# ## Overall trend for crimes in Denver

# In[ ]:


# mean and standard deviation of crimes per day
crimes_per_day = pd.DataFrame(crimes_df.resample('D').size())
crimes_per_day["MEAN"] = crimes_df.resample('D').size().mean()
crimes_per_day["STD"] = crimes_df.resample('D').size().std()
# upper control limit and lower control limit
UCL = crimes_per_day['MEAN'] + 3 * crimes_per_day['STD']
LCL = crimes_per_day['MEAN'] - 3 * crimes_per_day['STD']


# In[ ]:


plt.figure(figsize=(15,6))
df.resample('D').size().plot(label='Crimes per day', color='purple')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
crimes_per_day['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Overall trend of crimes in Denver', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of crimes')
plt.tick_params(labelsize=14)


# In[ ]:


month_df = crimes_df.resample('M').size()
plt.figure(figsize=(15,6))
month_df.plot(label='Total,  accidents per month', color='purple')
month_df.rolling(window=12).mean().plot(color='red', linewidth=5, label='12-Months Average')
plt.title('Overall trend of crimes in Denver(by month)', fontsize=16)


# In[ ]:


print("Best Month {0}: {1}".format(month_df.idxmin(), month_df[month_df.idxmin()]))
print("Worst Month {0}: {1}".format(month_df.idxmax(), month_df[month_df.idxmax()]))


# > - The overall trend seems to be increasing, and there's a slightly decrease in 2019
# > - But 2019 is not over and the violent month is not here yet
# > - We have more outliers toward the upper end (very violent days)
# > - Feburary 2014 is the best month with 4211 crimes
# > - August 2017 was the worst month with 6562 crimes

# ## Is the trend the same for all categories of crimes?

# In[ ]:


df.pivot_table(index='FIRST_OCCURRENCE_DATE', columns='OFFENSE_CATEGORY_ID', aggfunc='size', fill_value=0).resample('M').sum().rolling(window=12).mean().plot(figsize=(12,25), linewidth=4, cmap='inferno', subplots=True, layout=(-1, 3))
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# ## Most dangerous and safest neighborhoods

# In[ ]:


f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
neighborhood_freq = crimes_df.NEIGHBORHOOD_ID.value_counts()
dangerous = neighborhood_freq.iloc[:5]
safe = neighborhood_freq.iloc[-5:]
axes[0].set_title("Dangerous Neighborhoods")
sns.countplot(y="NEIGHBORHOOD_ID", data=crimes_df, order=dangerous.index, palette=p4, ax=axes[0])
axes[1].set_title("Safe Neighborhoods")
sns.countplot(y="NEIGHBORHOOD_ID", data=crimes_df, order=safe.index, palette=p4, ax=axes[1])


# ## Dangerous neighborhoods for ladies

# In[ ]:


sexual_assault_df = crimes_df[crimes_df.OFFENSE_CATEGORY_ID=="sexual-assault"]
f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
neighborhood_freq = sexual_assault_df.NEIGHBORHOOD_ID.value_counts()
dangerous = neighborhood_freq.iloc[:5]
safe = neighborhood_freq.iloc[-5:]
axes[0].set_title("Dangerous Neighborhoods")
sns.countplot(y="NEIGHBORHOOD_ID", data=sexual_assault_df, order=dangerous.index, palette=p4, ax=axes[0])
axes[1].set_title("Safe Neighborhoods")
sns.countplot(y="NEIGHBORHOOD_ID", data=sexual_assault_df, order=safe.index, palette=p4, ax=axes[1])


# ## Crimes by weekday

# In[ ]:


weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
week_df = pd.DataFrame(crimes_df["DAY_OF_WEEK"].value_counts()).sort_index()
week_df["DAY"] = weekdays
week_df.columns = ["Crime counts", "Week day"]
plt.figure(figsize=(8,6))
sns.barplot(x="Week day", y="Crime counts", color="purple", data=week_df)


# ## Average number of crimes per hour by category

# In[ ]:


crimes_hour_pt = df.pivot_table(index='OFFENSE_CATEGORY_ID', columns='HOUR', aggfunc='size')
crimes_hour_pt = crimes_hour_pt.apply(lambda x: x / crimes_hour_pt.max(axis=1))
plt.figure(figsize=(15,9))
plt.title('Average Number of Crimes per Hour by Category', fontsize=14)
sns.heatmap(crimes_hour_pt, cmap='inferno', cbar=True, annot=False, fmt=".0f");


# ## Average number of crimes per month by category

# In[ ]:


months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
crimes_pt = crimes_df.pivot_table(index='OFFENSE_CATEGORY_ID', columns='MONTH', aggfunc='size')
crimes_scaled = crimes_pt.apply(lambda x: x / crimes_pt.max(axis=1))
crimes_scaled.columns = months
plt.figure(figsize=(8,10))
plt.title('Average Number of Crimes per Category and Month', fontsize=14)
sns.heatmap(crimes_scaled, cmap='inferno', cbar=True, annot=False, fmt=".0f")


# ## Which day has the highest or lowest average number of crimes?

# In[ ]:


crimes_pt = crimes_df.pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=len)
crimes_pt_year_count = crimes_df.pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=lambda x: len(x.unique()))
crimes_avg = crimes_pt / crimes_pt_year_count
crimes_avg.columns = months
plt.figure(figsize=(10,12))
plt.title('Average Number of Complaints per Day and Month', fontsize=14)
sns.heatmap(crimes_avg.round(), cmap='inferno', linecolor='grey',linewidths=0.1, cbar=True, annot=True, fmt=".0f")


# > - Criminals love the first day of the month!
# > - And they take a break on Christmas Day

# In[ ]:


crimes_df = crimes_df.dropna(subset=['GEO_LAT', 'GEO_LON'])


# In[ ]:


robbery_df = crimes_df[(crimes_df.OFFENSE_CATEGORY_ID=='robbery') & (crimes_df.YEAR==2019)]


# In[ ]:


denver_map = folium.Map(location=[39.72378, -104.899157],
                       zoom_start=12,
                       tiles="CartoDB dark_matter")


# In[ ]:


for i in range(len(robbery_df)):
    lat = robbery_df.iloc[i]['GEO_LAT']
    long = robbery_df.iloc[i]['GEO_LON']
    popup_text = """Neighborhood: {}<br>
                    Date Occurred: {}<br>""".format(crimes_df.iloc[i]['NEIGHBORHOOD_ID'],
                                               crimes_df.iloc[i]['FIRST_OCCURRENCE_DATE'])
    folium.CircleMarker(location=[lat, long], popup=popup_text, radius=8, color='#800080', fill=True).add_to(denver_map)


# ## 2019 Denver Robbery Map

# In[ ]:


denver_map

