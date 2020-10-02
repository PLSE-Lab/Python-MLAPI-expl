#!/usr/bin/env python
# coding: utf-8

# (EDA) Noise Complaints in NYC 2018
# -------
# 

# ![](https://bloximages.chicago2.vip.townnews.com/oneidadispatch.com/content/tncms/assets/v3/editorial/8/04/804b0ff4-d4ac-5bcf-b501-9cb659c770cd/5ba25e8185125.image.jpg?resize=600%2C450)

# Introduction
# -------
# High volume of noise complaints are reported in New York City. 
# 
# In this brief exploratory data analysis, we will explore our data to cover the questions below:
# 
# 1. What type of complaints are most frequent? 
# 2. What areas in NYC get the most noise complaints?
# 3. When do most complaints occur?
# 
# *Data has been extracted from NYC Open Data*

# In[ ]:


#import libraries

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import plotly.express as px


# In[ ]:


#read dataset
noise_complaints_data = pd.read_csv('../input/Noise_Complaints.csv')


# In[ ]:


#preview data
noise_complaints_data.head()


# # Basic Information & Cleaning

# Let's take a look at our data. We immediately notice that we have a large set of columns (39 in total)

# | Feature | Description |
# | --- | --- |
# | Unique Key | Unique ID of the reported complaint case |
# | Created Date | The date that complaint was reported |
# | Closed Date | The date that the case was closed |
# | Agency  | Agency that handled the situation |
# | Agency Name | Full name of the agency that handled the situation |
# | Complaint Type | Type of complaint reported |
# | Descriptor | Description of the complaint |
# | Location Type | Type of location where the complaint was reported |
# | Incident Zip | Zip code of where the complaint was reported |
# | Incident Address | Mail box number and Street Address of where the complaint was reported |
# | Street Name | Street name of where the complaint was reported |
# | Cross Street 1 | Cross Street 1 of where the complaint was reported |
# | Cross Street 2 | Cross Street 2 of where the complaint was reported |
# | Intersection Street 1 | Intersection Street 1 of where the complaint was reported |
# | Intersection Street 2 | Intersection Street 2 of where the complaint was reported |
# | Address Type | Type of address of where the complaint was reported |
# | City | City address of where the complaint was reported |
# | Landmark | Landmark name of where the complaint was reported (if reported from a specific landmark) |
# | Facility Type | Facility type of where the complaint was reported |
# | Status | Status of the complaint (open/closed) |
# | Due Date | Due date of the complaint |
# | Resolution Description | How the complaint was handled by the agency |
# | Resolution Action Updated Date | Date the action has been taken regarding the complaint |
# | Community Board | Community Board of where the complaint took place |
# | Borough | Borough in NYC where complaint was reported |
# | X Coordinate (State Plane) | X coordinate of where the complaint was reported |
# | Y Coordinate (State Plane) | Y coordinate of where the complaint was reported |
# | Park Facility Name | Park facility name of where the complaint was reported |
# | Park Borough | Park Borough of the complaint |
# | Vehicle Type | N/A; Does not apply to this dataset |
# | Taxi Company Borough | N/A; Does not apply to this dataset |
# | Taxi Pick Up Location | N/A; Does not apply to this dataset |
# | Bridge Highway Name | N/A; Does not apply to this dataset |
# | Bridge Highway Direction | N/A; Does not apply to this dataset |
# | Road Ramp | N/A; Does not apply to this dataset |
# | Bridge Highway Segment | N/A; Does not apply to this dataset |
# | Latitude | Latitude coordinate of where the complaint was reported |
# | Longitude | Longitude coordinate of where the complaint was reported |
# | Location | Latitude and Longitude coordinate point of where the complaint was reported |

# In[ ]:


#show basic information of the dataset
noise_complaints_data.info()


# In[ ]:


#count missing items
noise_complaints_data.isna().sum()


# In[ ]:


#plot null values on a heatmap

sns.heatmap(noise_complaints_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# There many features that either have too many missing values or have repetitive information (many of which are related to location of the complaint) 
# 
# We will drop these columns along with the others that we will not be needing to answer our questions

# In[ ]:


#drop unneccessary columns
noise_complaints_data= noise_complaints_data.drop(['Status','Due Date','Agency','Agency Name','Landmark','Facility Type','Status','Due Date',
            'Resolution Description','Community Board','Park Facility Name','Park Borough','Vehicle Type',
            'Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp','Bridge Highway Segment'],axis=1)


# # 1) What type of complaints are most frequent? 

# Here we will answer our first question by checking out the most frequent complaint description and types

# **By Complaint Description**

# In[ ]:


#count complaints by description
comp_desc = (noise_complaints_data['Descriptor'].value_counts())
comp_desc


# From partying, car noise, construction, barking dogs, jack hammering, and more, we notice a great variety of complaint descriptions listed. 
# It is surprising to see that many of these descriptions were generalized into codes (NM1, NC1, etc.)

# In[ ]:


fig = go.Figure(data=[go.Histogram(y=noise_complaints_data['Descriptor'])])

fig.update_layout(
    autosize=False,
    width=800,
    height=800, 
    title=go.layout.Title(text="NYC Noise Complaint Descriptors 2018"))


# Although there were some interesting complaint descriptors, it seems that a great majority of these complaints were due to loud music, partying, banging, pounding, and loud talking. 
# 
# In fact, since we have 436,691 individual records in our dataset, the 200,000+ loud music/party complaint descriptor covers over half of these records. 

# **By Complaint Type**

# The main difference between complaint descriptor and complaint type is that complaint type is much more generalized. 

# In[ ]:


#view count of different kinds of complaints reported
comp_type = (noise_complaints_data['Complaint Type'].value_counts())
comp_type


# we are unfortunately stuck with a type that just says "Noise" without any other specification 

# In[ ]:


sns.countplot(y='Complaint Type',
              data=noise_complaints_data,
              order= noise_complaints_data['Complaint Type'].value_counts().index).set_title('NYC Noise Complaint Types 2018')


# we are able to notice immediately that residential noise complaints top this list, having over 200,000 records. 
# Once again, since we have 436,691 individual records, this means that over half of the them were due to residential complaints. 

# # 2) What areas in NYC get the most noise complaints?

# With the information we have we are able to categorize and locate the number of complaints by area. 
# 
# However, before we continue, one key assumption to remember is that the area with high noise complaints DOES NOT neccessarily mean that it is the noisest area
# 
# To explore this question, the focus will be on the **Borough**, **City**, and **Location Type**

# ## By Borough

# In[ ]:


#find unique borough names
noise_complaints_data['Borough'].unique()


# In[ ]:


#group count of complaints by borough
count_borough = noise_complaints_data['Borough'].value_counts()
count_borough


# In[ ]:


#create pie chart
fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(aspect='equal'))

borough = list(count_borough.index)
count = list(count_borough)

def func(pct):
    absolute = int(pct/100.*np.sum(count))
    return "{:.1f}% ({:d})".format(pct, absolute)


ax.pie(count_borough, 
       autopct=lambda pct: func(pct),
      pctdistance=1.2)

ax.legend(borough,
          title="Borough",
          loc="center left",
          bbox_to_anchor=(1, 0.3, 0.5, 1))

plt.title('NYC Noise Complaints by Borough 2018')


# Of the five boroughs, Manhattan and Brooklyn take the top with Bronx and Queens coming next. Staten Island only takes a small proportion
# 
# 0.2% (1090) unspecified items may be a combination of unidentified boroughs that have not been documented for whatever reason

# ## By City

# In[ ]:


noise_complaints_data['City'].unique()


# In[ ]:


comp_city = (noise_complaints_data['City'].value_counts())


# In[ ]:


comp_city_df = comp_city.to_frame().reset_index()
comp_city_df.rename(columns={'index':'City Name'},inplace=True)
comp_city_df.rename(columns={'City':'Count'},inplace=True)
comp_city_df.head(10)


# In[ ]:


sns.barplot(x='Count',
            y='City Name',
            data=comp_city_df.head(10)).set_title('Top 10 NYC Noise Compaints by City 2018')


# Similar information can be found as when we explored count of complaints by borough. 
# 
# However, it seems that Queens is broken down into a great variety of city names

# ## By Location Type

# In[ ]:


noise_complaints_data['Location Type'].value_counts()


# In[ ]:


sns.countplot(y='Location Type',
              data=noise_complaints_data,
              order= noise_complaints_data['Location Type'].value_counts().index).set_title('NYC Noise Complaints by Location Type 2018')


# Complaints from residential areas account for over half of the noise complaints

# ## Plot Map

# In an attempt to identify geographical patterns of reported complaints, we will use the longitude and latitude points to create rough plots

# In[ ]:


#find maximum and minimum longitude
print(noise_complaints_data['Longitude'].max())
print(noise_complaints_data['Longitude'].min())


# In[ ]:


#find maximum and minimum latitude
print(noise_complaints_data['Latitude'].max())
print(noise_complaints_data['Latitude'].min())


# In[ ]:


#plot complaints

#resize plot
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

plt.plot(noise_complaints_data['Longitude'],noise_complaints_data['Latitude'],'.',markersize=0.2)
plt.title('NYC Noise Complaints 2018')


# Each point in this plot is a reported noise complaint. Darker spots of the plot indicate high volume of complaints. 
# We can almost see a visible map of New York City with Manhattan and Brooklyn having the most defined outline. 
# 
# Let's take a closer look at the top 4 boroughs

# In[ ]:


#closer look at Manhattan, Brooklyn, Bronx, Queens

#resize plot
from pylab import rcParams
rcParams['figure.figsize'] = 30, 13

#create subplot 
fig, axes = plt.subplots(nrows=1,ncols=4)

#filter boroughs
noise_complaints_manhattan= noise_complaints_data[noise_complaints_data['Borough']=='MANHATTAN']
noise_complaints_brooklyn= noise_complaints_data[noise_complaints_data['Borough']=='BROOKLYN']
noise_complaints_bronx= noise_complaints_data[noise_complaints_data['Borough']=='BRONX']
noise_complaints_queens= noise_complaints_data[noise_complaints_data['Borough']=='QUEENS']

#plot
axes[0].plot(noise_complaints_manhattan['Longitude'],noise_complaints_manhattan['Latitude'],'.',markersize=0.6)
axes[0].set_title('Manhattan Noise Complaints 2018')

axes[1].plot(noise_complaints_brooklyn['Longitude'],noise_complaints_brooklyn['Latitude'],'.',markersize=0.6)
axes[1].set_title('Brooklyn Noise Complaints 2018')

axes[2].plot(noise_complaints_bronx['Longitude'],noise_complaints_bronx['Latitude'],'.',markersize=0.6)
axes[2].set_title('Bronx Noise Complaints 2018')

axes[3].plot(noise_complaints_queens['Longitude'],noise_complaints_queens['Latitude'],'.',markersize=0.6)
axes[3].set_title('Queens Noise Complaints 2018')


# In[ ]:


#show geographical heatmap; show only 30000 datpoints due to memory limit

head = noise_complaints_data.head(30000)

import folium
from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(head[['Latitude','Longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)


# With each of the four boroughs broken down, we can see that noise complaints cluster around certain neighborhoods
# 
# For example, in Manhattan, there seems to be a lot of noise complaints around East Village and Lower East side. We also see dark shades in the area above central park (Harlem, Washington Heights, Fort George, etc). 
# 
# Meanwhile, it's interesting to note that areas of midtown which include popular tourist attractions such as Times Square and Rockfeller Center do not have as much complaints reported as other zones in the city. This may be due to the fact that these zones are not typical residential areas (thus less residential complaints)
# 
# For the other boroughs, we see that the dark shades are areas geographically close to Manhattan (Williamsburg for Brooklyn, Long Island City for Queens) 
# 

# In[ ]:


#find maximum and minimum longitude
print(noise_complaints_data['Latitude'].mean())
print(noise_complaints_data['Longitude'].mean())


# # 3) When do most complaints occur?

# In this section, we will explore to see if there are patterns in the volume of noise complaints by hour, day of the week, and by month. 
# 
# However, we must first prepare our date/time data to fit our needs

# In[ ]:


#check the current created date format
type(noise_complaints_data['Created Date'].iloc[0])


# The created date column is currently in a string format so we should convert it to a date/time format for our convenience

# In[ ]:


#convert created date to date time format
noise_complaints_data['Created Date'] = pd.to_datetime(noise_complaints_data['Created Date'])


# In[ ]:


#create hour, month, day of week columns
noise_complaints_data['Hour'] = noise_complaints_data['Created Date'].apply(lambda time: time.hour)
noise_complaints_data['Month'] = noise_complaints_data['Created Date'].apply(lambda time: time.month)
noise_complaints_data['Day of Week'] = noise_complaints_data['Created Date'].apply(lambda time: time.dayofweek)


# In[ ]:


#check format of the day of the week column
noise_complaints_data['Day of Week'].head()


# The day of the week column is currently set as integer values (0 = Monday, 1 = Tuesday,... 6 = Sunday) 
# 
# This is completely fine but we'll convert this to string values that we are more familar to seeing

# In[ ]:


#convert day of the week from integer to string descriptions
d_o_w = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
noise_complaints_data['Day of Week'] = noise_complaints_data['Day of Week'].map(d_o_w)


# **By Hour**

# In[ ]:


#plot count of complaints by time

#resize plot
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

sns.countplot(x=noise_complaints_data['Hour'])
plt.title('NYC Count of Complaints 2018: by Hour')


# The complaints are counted by military hour. It makes sense that a large volume of these complaints were reported at night time (with high spikes around midnight) 

# **By Day of Week**

# In[ ]:


#plot count of complaints by day of week

sns.countplot(x=noise_complaints_data['Day of Week'])
plt.title('NYC Count of Complaints 2018: by Day of Week')


# This time the complaints are counted by day of week. From this visualization, we may infer that higher volume of complaints are recorded during weekends compared to weekdays.

# **By Month**

# In[ ]:


#plot count of complaints by month

sns.countplot(x=noise_complaints_data['Month'])
plt.title('NYC Count of Complaints 2018: by Month')


# The following count plot shows us a potential seasonal trend when it comes to noise complaints in NYC. **Summer** months such as *May*, *June*, *July* have high records of complaints while **winter** months such as *November*, *December*, *January*, and *February* have lower counts compared to other months. 
# 
# Meanwhile, **fall** months such as *September* and *October* were also very hot for noise complaints

# **By Date**

# In[ ]:


noise_complaints_data['Date'] = noise_complaints_data['Created Date'].apply(lambda t:t.date())


# In[ ]:


noise_complaints_data.groupby(noise_complaints_data['Date']).count()['Unique Key'].plot()
plt.tight_layout()
plt.title('NYC Count of Complaints 2018: by Date')


# This visualization replicates information from the previous plot but with a time series-like presentation. 
# 
# Multiple factors that we have already looked at (seasonality, time of day, weekend/weekday) may have impacted the up and down spikes. 

# **Date/Time Relationships**

# Here we will explore relationships between our date/time variables for count of noise complaints

# In[ ]:


#create data frame of count of complaints on day of week vs hour
dayHour = noise_complaints_data.groupby(by=['Day of Week','Hour']).count()['Unique Key'].unstack()
dayHour


# In[ ]:


#create heatmap of count of complaints on day of week vs hour
sns.heatmap(data=dayHour,cmap='coolwarm')
plt.title('NYC Count of Complaints 2018: Day of Week vs. Hour')


# We already noted that large volume of complaints were recorded during weekends and night time and this heatmap confirms just that. 
# 
# It also seems that lower number of noise complaints were recorded during weekday business hours

# In[ ]:


#create data frame of count of complaints on month vs hour
monthDay = noise_complaints_data.groupby(by=['Month','Day of Week']).count()['Unique Key'].unstack()
monthDay


# In[ ]:


#create heatmap of count of complaints on month vs hour
sns.heatmap(data=monthDay,cmap='coolwarm')
plt.title('NYC Count of Complaints 2018: Day of Week vs. Month')


# Here we see a possible combination of seasonality and weekday/weekend being factors. 
# 
# Weekends during Spring, Summer, and Fall months recorded more complaints than other times.  

# **Conclusion**
# 
# This marks the end of this notebook!
# 
# Thanks for reading and feel free to leave any comments and suggestions regarding this EDA
