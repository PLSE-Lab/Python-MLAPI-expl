#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[ ]:


#Reading the dataset into pandas
df=pd.read_csv("/kaggle/input/uncover/UNCOVER/apple_mobility_trends/mobility-trends.csv")


# The above dataset contains data about relative volume of Direction Requests from differen region in Apple Maps. The Data of Jan 13 2020 has been fixed as 100 and for the succeding dates it is the relative volume as compared with Jan 13th data.

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.dropna(subset=['date'],axis='rows',inplace=True)
df['pdate']=pd.to_datetime(df.date,format="%Y-%m-%d")
df["dayofweek"]=df.pdate.dt.dayofweek
#df.date=df.date.dt.strftime("%Y-%m-%d")
gr_region=df.groupby("region")
gr_date=df.groupby("date")
df_region=gr_region.sum()
df_date=gr_date.sum()
df_india=df[df.region=='India']


# In[ ]:



df_date.head()
df.head()
#df_india.transportation_type.unique()


# # Line plot for seeing the trend of Direction reuqests globally for the given period.

# In[ ]:


plt.figure(figsize=(24,10))
plot1=sns.lineplot(x=df_date.index,y='value',data=df_date)
plot1.set_xticklabels(df_date.index,rotation=90)
plot1.set_title("Trend of Direction request")


# # Line plot of Direction Request Trend in India
# A Simple line plot of relative volume of Direction Requests from India.
# 

# In[ ]:


plt.figure(figsize=(24,10))
plot2=sns.lineplot(x='date',y='value',data=df_india[df['transportation_type']=='driving'],legend='brief',label='Driving')
sns.lineplot(x='date',y='value',data=df_india[df['transportation_type']=='walking'],legend='brief',label='Walking')
plot2.set_xticklabels(df_india[df_india['transportation_type']=="driving"].date,rotation=90)
plot2.set_title("Trend of Direction request in India",fontsize=30)
plot2.set_xlabel('Date',fontsize=20)
plot2.set_ylabel('Relative Volume of Direction Requests',fontsize=20)


# # Observation
# 1. An immidiately visible observation is that the number of requests made fell rapidly on 22 march(Janta Curfew), it saw a short spike next day followed by a drop which is consistent during the lockdown announced on 24 March.
# 
# 2. Yet another observation is that, after the lockdown, Direction request made by walkers is markedly and consistenly more than the direction rquest made by Drivers. This is in line with expectation due to restriction of vehicular movement, long distance travel, etc.
# 
# 3. Prior to lockdown we can also see a recurring trend of peaking repating at seemingly same intrevals. Further analysis required.

# In[ ]:


df_india.dtypes


# In[ ]:


df.dayofweek
df.region.unique().size
df[df['date']=='2020-01-13']


# # Line plot of request trend against Day of Week
# ### Peaking seems to be occuring at a weekly intreval, so plotting across day of week to determine if peaking happen on any particular day.

# In[ ]:


plt.figure(figsize=(24,20))
plot3=sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date<='2020-03-23'],legend='brief',label='Before Lockdown')
sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date>'2020-03-23'],legend='brief',label='During Lockdown')
xlabels={'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'}
plot3.set_xticklabels(xlabels,rotation=90)
plot3.set_title("Driection Request Trend against Day of Week",fontsize=30)
plot3.set_xlabel('Day of Week',fontsize=20)
plot3.set_ylabel('Relative Volume of Direction Requests',fontsize=20)

#sns.lineplot(x='dayofweek',estimator='sum',y='value',data=df_india[df_india.date>'2020-03-22'],legend='brief',label='During Lock down')


# ## Observation
# 1. It is evident peaking happend on saturdays prior to lockdown. The reason might be as people tend to go on long drives, new places on saturday and tend to use maps more as compared to other days when the usual reason for travel is office. Sunday is comparitively lower than saturday but still greater than wekdays.
# 2. During lockdown the line is almost flat with no peaks as expected.

# In[ ]:


#A function to mark a date as ALS(After Lockdown Start, or BLS - before lockdown start) for engineering categorical variable.
def markPeriod(dt):
    if(dt>'2020-03-23'):
        return "ALS" #After Lockdown Start
    else:
        return "BLS"  #Before Lockdown Start
#Creating a categorical variable
df_india['period']=df_india.apply(lambda x : markPeriod(x['date']),axis=1)


# # Plotting boxplots of request volume against  categorial variable Transportation Type categorical variable.

# In[ ]:


plt.figure(figsize=(24,20))
plot4=sns.boxplot(x='transportation_type',y='value',data=df_india[df_india.period=='ALS'], notch=True)
plot4.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot4.set_title("Boxplot of Transportation Type",fontsize=30)
plot4.set_xlabel('Transportation Type',fontsize=20)
plot4.set_ylabel('Relative Volume of Direction Requests',fontsize=20)


# ## Observations:
# 1. The boxplots are skewed to the right as is evident from the top whiskers being long.
# 2. No outliers found for the above plots.
# 3. Mean and median of walkers is significantly higher than mean for drivers as is seen from the non overlap of the notches.

# # Plotting boxplots of request volume against  categorial variable Before/After Lockdown period.

# In[ ]:


plt.figure(figsize=(24,20))
plot5=sns.boxplot(x='period',y='value',data=df_india, notch=True)
plot5.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot5.set_title("Boxplot of Time Period",fontsize=30)
plot5.set_xlabel('Before/After Lockdown',fontsize=20)
plot5.set_ylabel('Relative Volume of Direction Requests',fontsize=20)


# ## Observations:
# 1. The boxplots look symmetrical.
# 2. More outliers on the lower side of the boxplot for Before Lockdown start, this might be due to the fact than restriction were  starting to come in place before the official start of lockdown.
# 3. Again Mean and median of BLS is significantly higher than mean, median for ALS as is seen from the non overlap of the notches. This is as expected and observed from previous line plots.

# In[ ]:


plt.figure(figsize=(24,20))
sns.distplot(df[df['region']=='India'].value)
plot5.set_xticklabels(xlabels,rotation=90,fontsize=15)
plot5.set_title("Boxplot of Time Period",fontsize=30)
plot5.set_xlabel('Before/After Lockdown',fontsize=20)
plot5.set_ylabel('Relative Volume of Direction Requests',fontsize=20)


# In[ ]:


import folium
from folium import plugins
import json
import os


# In[ ]:


json_path = os.path.join(os.getcwd(),'mydataset/','countries.geojson') 
world_geo = json.load(open("/kaggle/input/my-input-data/countries.geojson"))


# In[ ]:


#Code to try assign countries for cities, and subregions.
#code takes lots of time to run, need to try for alternative machanism.
df_citymap=pd.read_csv("/kaggle/input/my-input-data/world-cities_csv.csv")
def getCountry(city):
    #print(city)
    country=df_citymap[df_citymap.country==city].country.unique()
    if(country.size!=0):
        return country[0]
    country=df_citymap[df_citymap.subcountry==city].country.unique()
    if(country.size!=0):
         return country[0]
    country = df_citymap[df_citymap.name==city].country
    return "".join(country.tail(1))
    return city

#print(getCountry('Chennai'))
#df_temp=df.copy()
#df_temp['country']=df_temp.apply(lambda x: getCountry(x['region']),axis=1)
#df_temp.head()


# In[ ]:


df['period']=df.apply(lambda x : markPeriod(x['date']),axis=1)
df_countrywise_BLS=df[df.period=='BLS']
df_countrywise_ALS=df[df.period=='ALS']
df_countrywise_BLS=df_countrywise_BLS[(df_countrywise_BLS.geo_type=='country/region')|(df_countrywise_BLS.geo_type=='sub-region')].groupby('region').sum()
df_countrywise_ALS=df_countrywise_ALS[(df_countrywise_ALS.geo_type=='country/region')|(df_countrywise_ALS.geo_type=='sub-region')].groupby('region').sum()
df_countrywise_BLS.head()


# In[ ]:


#Below is a word map overlayed with data about cumulative Direction requests in intial days of COVID.

world_map_BLS = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map_BLS.choropleth(
    geo_data=world_geo,
    data=df_countrywise_BLS,
    columns=[df_countrywise_BLS.index, 'value'],
    key_on='feature.properties.ADMIN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Direction Requests'
)
world_map_BLS


# In[ ]:


#Below is a word map overlayed with data about cumulative Direction requests of recent past.

world_map_ALS = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map_ALS.choropleth(
    geo_data=world_geo,
    data=df_countrywise_ALS,
    columns=[df_countrywise_ALS.index, 'value'],
    key_on='feature.properties.ADMIN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Direction Requests'
)
world_map_ALS

