#!/usr/bin/env python
# coding: utf-8

# # Crimes in Boston

# Crimes in Boston is a dataset from Kaggle ( https://www.kaggle.com/AnalyzeBoston/crimes-in-boston/downloads/crimes-in-boston.zip/2 ) that I found very useful in practicing data exploratory and visualisation skills. Doing my 'exercise' I found two very inspiring notebooks which showed this data in a very convinient way:
# https://www.kaggle.com/heesoo37/boston-crimes-starter-eda
# https://www.kaggle.com/frankkloster/bostom-crimes-eda
# Please do have a look at them cause I won't repeat all the steps done there. As I explored the dataset on my own I hit some new ideas which I would like to share with you :)

# ## Looking at data

# First let's read in some libraries. Then load our data and explore them a little bit.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/crime-cleanedcsv/crime.csv')
df.head()


# Check and eventually remove duplicates.

# In[ ]:


print(df.shape, df.drop_duplicates().shape)
df = df.drop_duplicates()


# We had 23 duplicates. Now we would like to know what type of columns do we have. This is also a quick check for lacking values.

# In[ ]:


df.info()


# In[ ]:


df.describe()


# There is an obvious lack of "SHOOTING" values and also some missing values in "STREET", "Lat", "Long". Despite that we also know that data are from Boston so "Lat" & "Long" should have litte std and above we see some "-1"s... Let's clean that.

# In[ ]:


df["SHOOTING"].fillna("N", inplace = True)
df["DAY_OF_WEEK"] = pd.Categorical(df["DAY_OF_WEEK"], 
              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
              ordered=True)
df["Lat"].replace(-1, None, inplace=True)
df["Long"].replace(-1, None, inplace=True)

df.describe()


# Now std seems about right. I would like to add some more data columns for further visualisation. 
# We have date in format 'YYYY-mm-DD HH:MM:SS', 'YYYY', 'mm', but it could be also useful to have it in 'YYYY-mm-DD' and 'YYYY-mm'.

# In[ ]:


def getDate(dateStr, numChar):
    return dateStr[0:numChar]

df['DATE'] = df['OCCURRED_ON_DATE'].apply(getDate, numChar = 10)
df['YEARMONTH'] = df['OCCURRED_ON_DATE'].apply(getDate, numChar = 7)

df[['YEARMONTH', 'DATE', 'OCCURRED_ON_DATE']].head()


# ## Why this analysis differ?

# Two best analyses of this dataset which I mentioned on the beggining of this notebook made an assumption that the records in our dataset are unique. They are. Howerev in my opinion crimes are NOT.

# There is a column 'INCIDENT_NUMBER' having a high potential of being an identity number of an event. Considering that each crime has it's own ID we have some duplicates... How will it change our analysis and conclusions?

# ### Why duplicated?

# In[ ]:


print('Num of records: {} \nNum of events: {}'.format(df.shape[0], df["INCIDENT_NUMBER"].nunique()))


# In[ ]:


print("It  seemes like there are {} records in database per 1 crime.".format(round(df.shape[0]/df["INCIDENT_NUMBER"].nunique(),2)))


# Maybe one crime can be assigned to different classes?

# In[ ]:


tmp = df.groupby("INCIDENT_NUMBER")["YEAR"].count().sort_values(ascending = False)
tmp.head(10)


# We can see that there are crimes that have over 10 records. Max of it is 13. Let's make a summary:

# In[ ]:


tmp.value_counts() #Index: num of records per crime, Values: num of occurences of such a case.


# In[ ]:


print('It occurs that {}% of our events are "duplicated" at least 2 times.'.format(round(100*(282517 - 254996) / 282517),2))


# Let's see what is inside two top "complicated" crimes.

# In[ ]:


df[df["INCIDENT_NUMBER"] == "I162030584"]


# In[ ]:


df[df["INCIDENT_NUMBER"] == "I152080623"]


# Looking into details we can discover that one crime can have different UCR_PART and different offence categorization (OFFENSE_CODE, OFFENSE_DESCRIPTION), but duplicated OFFENSE_CODE_GROUP (eg. rows with index 290817, 290818). This can bring some significant changes in visualisations and interpreting. Thus it would be good to carefully deduplicate records depending on questions we would like to answear.

# ### Crimes in time

# Let's see if something changes in relations with time occurences.

# In[ ]:


timeOccurencesNormal = df[['INCIDENT_NUMBER','OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'SHOOTING',
                           'DAY_OF_WEEK', 'HOUR', 'DATE', 'YEARMONTH']]
timeOccurencesDedup  = df[['INCIDENT_NUMBER','OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'SHOOTING',
                           'DAY_OF_WEEK', 'HOUR', 'DATE', 'YEARMONTH']].drop_duplicates()

print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), timeOccurencesDedup.shape[0]))


# In[ ]:


fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (18, 10))

sns.countplot(timeOccurencesNormal["YEAR"], color='lightblue', ax = axes[0,0] )
axes[0,0].set_title("Number of crimes")
sns.countplot(timeOccurencesNormal["DAY_OF_WEEK"], color='lightgreen', ax = axes[1,0])
axes[1,0].set_title("Number of crimes")
sns.countplot(timeOccurencesNormal["HOUR"], color = 'orange', ax = axes[2,0])
axes[2,0].set_title("Number of crimes")

sns.countplot(timeOccurencesDedup["YEAR"], color='lightblue', ax = axes[0,1] )
axes[0,1].set_title("Number of crimes (deduplicated)")
sns.countplot(timeOccurencesDedup["DAY_OF_WEEK"], color='lightgreen', ax = axes[1,1] )
axes[1,1].set_title("Number of crimes (deduplicated)")
sns.countplot(timeOccurencesDedup["HOUR"], color = 'orange', ax = axes[2,1] )
axes[2,1].set_title("Number of crimes (deduplicated)")

plt.tight_layout()


# It seems to change nothing but the scale of a plot. So there would be a difference in reporting, but conslusions about trends would stay the same.

# Next I would like to look at dependencies between weekdays, hours and number of cases reported (this is something I haven't found in previous notebooks).

# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))

sns.heatmap(pd.pivot_table(data = timeOccurencesNormal, index = "DAY_OF_WEEK", 
                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count'), 
               cmap = 'Reds', ax = axes[0])
sns.heatmap(pd.pivot_table(data = timeOccurencesDedup, index = "DAY_OF_WEEK", 
                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count')
               , cmap = 'Reds', ax = axes[1])


# There is a difference between working days and weekends. Whereas from Monday to Friday a period of "lighter" hours with fewer crimes spans from 1/2AM - 6/7AM, in Weekends it seems to be shifted to 3AM - 7/8AM (due to partying? ;) ).  There is also a buch of very "busy" hours (4PM - 6PM) during week which does not occur on Saturday and Sunday (stressed office workers comming out on the streets?). However still comparison between old and deduplicated version of data didn't bring any differences.

# Are there any time trends in number of crimes at all?

# In[ ]:


fig = plt.figure(figsize=(18,6))

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(timeOccurencesNormal.groupby('DATE').count(), 
          c = 'blue', label = "Original data")
axes.plot(timeOccurencesDedup.groupby('DATE').count(), 
          c = 'green', label = "Dedup data")
plt.xticks(rotation = 90)
plt.legend()
axes.set_title("Number of crimes in a day")
axes.set_ylabel("Number of crimes")

labelsX = timeOccurencesNormal.groupby('DATE').count().index[::30]
plt.xticks(labelsX, rotation='vertical')

#I've got duplicated legend here, so I used remedy:
# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
handles, labels = axes.get_legend_handles_labels() 
i = np.arange(len(labels))
filter = np.array([])
unique_labels = list(set(labels))
for ul in unique_labels:
    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 
    
handles = [handles[int(f)] for f in filter] 
labels = [labels[int(f)] for f in filter]
axes.legend(handles, labels) 


# In[ ]:


fig = plt.figure(figsize=(18,6))

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(timeOccurencesNormal.groupby('YEARMONTH').count(), 
          c = 'blue', label = "Original data")
axes.plot(timeOccurencesDedup.groupby('YEARMONTH').count(), 
          c = 'green', label = "Dedup data")
plt.xticks(rotation = 90)
plt.legend()
axes.set_title("Number of crimes in a month")
axes.set_ylabel("Number of crimes")

#I've got duplicated legend here, so I used remedy:
# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
handles, labels = axes.get_legend_handles_labels() 
i = np.arange(len(labels))
filter = np.array([])
unique_labels = list(set(labels))
for ul in unique_labels:
    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 
    
handles = [handles[int(f)] for f in filter] 
labels = [labels[int(f)] for f in filter]
axes.legend(handles, labels)  


# Again we come to the situation where mainly number of events decreased after deduplication, but trends didn't change. 
# On the other hand the second plot shows that we have summer hypes of crimes and significant drop in their rate on the beggining of the year - and it is not a trend in one year only. For me it was more obvious way of seeing that than looking on bar plots aggregated by month without a year information or looking on noisy day-by-day plot above.

# If deduplication does not change "anything" in overall crimes seen in dimension of time maybe going into details will be more usefull? Let's dig into shooting crimes.

# In[ ]:


print('We have {}% of shooting crimes in all events (deduplicated situation).'.format(
    round(100*timeOccurencesDedup[timeOccurencesDedup['SHOOTING'] == 'Y'].shape[0]/timeOccurencesDedup.shape[0],2)))


# In[ ]:


fig = plt.figure(figsize=(18,6))

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('DATE').count(), 
          c = 'lightblue', label = "Original data")
axes.plot(timeOccurencesDedup[timeOccurencesDedup["SHOOTING"] == "Y"].groupby('DATE').count(), 
          c = 'black', label = "Dedup data")
plt.xticks(rotation = 90)
plt.legend()
axes.set_title("Shooting crimes")
axes.set_ylabel("Number of crimes with shooting")

labelsX = timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('DATE').count().index[::30]
plt.xticks(labelsX, rotation='vertical')

#I've got duplicated legend here, so I used remedy:
# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
handles, labels = axes.get_legend_handles_labels() 
i = np.arange(len(labels))
filter = np.array([])
unique_labels = list(set(labels))
for ul in unique_labels:
    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 
    
handles = [handles[int(f)] for f in filter] 
labels = [labels[int(f)] for f in filter]
axes.legend(handles, labels) 


# In[ ]:


fig = plt.figure(figsize=(18,6))

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('YEARMONTH').count(), 
          c = 'blue', label = "Original data", marker = "o")
axes.plot(timeOccurencesDedup[timeOccurencesDedup["SHOOTING"] == "Y"].groupby('YEARMONTH').count(), 
          c = 'green', label = "Dedup data", marker="o")
plt.xticks(rotation = 90)
plt.legend()
axes.set_title("Shooting crimes")
axes.set_ylabel("Number of crimes with shooting")

#I've got duplicated legend here, so I used remedy:
# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
handles, labels = axes.get_legend_handles_labels() 
i = np.arange(len(labels))
filter = np.array([])
unique_labels = list(set(labels))
for ul in unique_labels:
    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 
    
handles = [handles[int(f)] for f in filter] 
labels = [labels[int(f)] for f in filter]
axes.legend(handles, labels) 


# This time also plot day-by-day is very "lousy" and the main conclusion which I took from it is that there was maximum 4 shootings in one day instead of 10. But I would like to move to plot by 'YEARMONTH' which is a twist in our analysis. We can see that after deduplication trends has changed, especially in the 2018, where 'duplicated' analysis would report that number of shootings is decreasing where in fact it will be slowly increasing. Also on a blue line we have a maximum peak in December 2017 whereas the true one is in July 2015...

# So when in the day is it better to have a gun? Are we free at weekends?

# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))

sns.heatmap(pd.pivot_table(data = timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"], index = "DAY_OF_WEEK", 
                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count'), 
               cmap = 'Reds', ax = axes[0])
sns.heatmap(pd.pivot_table(data = timeOccurencesDedup[timeOccurencesNormal["SHOOTING"] == "Y"], index = "DAY_OF_WEEK", 
                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count')
               , cmap = 'Reds', ax = axes[1])


# Ommiting a peak in Saturday morning (propably case of one day only) there is a slight trend of range of safer hours 2AM - 2PM in working days and again shifted 4AM - 2PM in Weekends, but it is not as clear as we've seen it in overall crimes. Also deduplication shows more 'outliers'.

# ### Crimes in space

# Let's move to space dimension now. We remember that at the first glance into our data we had some missing values in coordinates - it's good to mention here that those observations are ignored by plots. 

# In[ ]:


locationOccurencesNormal = df[['INCIDENT_NUMBER','DISTRICT', 'REPORTING_AREA', 'SHOOTING','Lat', 'Long']]
locationOccurencesDedup  = df[['INCIDENT_NUMBER','DISTRICT', 'REPORTING_AREA', 'SHOOTING','Lat', 'Long']].drop_duplicates()
print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), locationOccurencesDedup.shape[0]))


# How does security in districts looks like?

# In[ ]:


# Plot districts
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))
sns.scatterplot(y='Lat',
                x='Long',
                hue='DISTRICT',
                alpha=0.01,
                data=locationOccurencesNormal, 
                ax = axes[0])
#plt.ylim(locationOccurencesNormal['Long'].max(), locationOccurencesNormal['Long'].min())
sns.scatterplot(y='Lat',
                x='Long',
                hue='DISTRICT',
                alpha=0.01,
                data=locationOccurencesDedup, 
                ax = axes[1])


# As we can see there are some safer places. An interesting one is a blue district in the bottom of the plot seems to have only some main streets being very dangerous but not the rest - maybe there is a special structure of buildings (big companies, store houses etc.?). I will leave it for your future exploration. What is good to piont here is that the are no visible effects of deduplication. Let's go to shooting analysis to look for differences.

# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))
sns.scatterplot(y = 'Lat',
                x = 'Long',
                alpha = 0.3,
                data = locationOccurencesNormal[locationOccurencesNormal["SHOOTING"]=="Y"], 
                ax = axes[0])
axes[0].set_title("Crime locations")
sns.scatterplot(y = 'Lat',
                x = 'Long',
                alpha = 0.3,
                data = locationOccurencesDedup[locationOccurencesDedup["SHOOTING"]=="Y"], 
                ax = axes[1])
axes[1].set_title("Crime locations (deduplicated)")


# First and main conclusion is that there are far more shootings in the center of Boston than on the suburbs. But on the right plot those suburbs seem to be more safer ones, where shooting happens sporadically...

# But what about complicated crimes? Those that include many law violations. In time dimension we could see them as differences in numbers, but here with locations it is not so easily seen... So let's visualise them! 

# In[ ]:


# Below I choose a YEAR column cause I would like to narrow the data processed 
# and this columns is nice -> doesn't have any null values 
tmp = df.groupby('INCIDENT_NUMBER')['YEAR'].count().sort_values(ascending = False)
tmp = pd.DataFrame({'INCIDENT_NUMBER': tmp.index, 'NUM_RECORDS': tmp.values})
seriousCrimes = df.merge(tmp[tmp['NUM_RECORDS'] > 5], on = 'INCIDENT_NUMBER', how = 'inner')
seriousCrimes = seriousCrimes[['INCIDENT_NUMBER', 'Lat','Long']].drop_duplicates()[['Lat','Long']].dropna()


# In[ ]:


#!pip install folium
# Used this tutorial: https://medium.com/@bobhaffner/folium-markerclusters-and-fastmarkerclusters-1e03b01cb7b1
import folium
from folium.plugins import MarkerCluster

some_map = folium.Map(location = [seriousCrimes['Lat'].mean(), 
                                  seriousCrimes['Long'].mean()], 
                      zoom_start = 12)
mc = MarkerCluster()
#creating a Marker for each point. 
for row in seriousCrimes.itertuples():
    mc.add_child(folium.Marker(location = [row.Lat,  row.Long]))

some_map.add_child(mc)

some_map


# Most complex crimes - not suprisingly - happen in the center of Boston...

# ### Join time and space

# Is there anything interesting we can get from joining time with space?

# In[ ]:


locationTimeOccurencesDedup = df[['INCIDENT_NUMBER', 'SHOOTING','Lat', 'Long', 'DAY_OF_WEEK', 'HOUR']].drop_duplicates()
print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), locationOccurencesDedup.shape[0]))


# In[ ]:


fig = plt.figure(figsize=(18,10))
g = sns.FacetGrid(data = locationTimeOccurencesDedup[(locationTimeOccurencesDedup['HOUR'] >= 1) & (locationTimeOccurencesDedup['HOUR'] <= 7)],
                                                  row = 'HOUR', col = 'DAY_OF_WEEK')
g = g.map(sns.scatterplot, 'Long', 'Lat', alpha=0.03)


# Preceding plots show crime locations in selected hour and day of week. I cut the hours to our 'peaceful' period to see if we have any dangerous districts during that time. I don't see any trend in that case. But we can notice that there are more dangerous spots (clubs?) for Weekends about 1-2 AM.

# ## We came to an end...

# This notebook is based on assumption that INCIDENT_NUMBER contains a unique ID of a crime (before usage of any of the conclusions please be aware to confirm that with the data owner). I found previous analysises very useful, inspiring and exhaustive. However I wanted to share my view on duplicated records usecase. Although it could change the analysis very impactful it mainly occured in drops of numbers on this dataset. It can have more impact when going into details - especially type of crimes (which I haven't have time to explore further - maybe will catch up in the future), so I hope that will inspire people to dig deeper. Those 'duplicates' also created new opportunities for dependency analysis of connected types of crimes (tried to perform affinity analysis, but too little cooccurences appeared... :( ). I hope this will come with continuation, too!

# Thank you for reading this notebook. I hope you liked it (if so please upvote!) and will get inspired to dig into data in your near future. Maybe you would like to explore this data set a little further? :)
# 
# If you have any comments or questions please find me via Kaggle :)

# In[ ]:




